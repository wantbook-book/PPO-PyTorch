#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于多进程的VllmSampler实现

解决原有MultiVllmSampler的问题：
1. 避免Python GIL限制
2. 每个进程独立管理GPU资源
3. 更好的内存隔离
4. 避免CUDA上下文冲突
"""

from vllm import LLM, SamplingParams
from sae_lens import SAE
from utils.sae_utils import add_hooks, get_multi_intervention_hook
import torch
from functools import partial
from transformers import AutoModelForCausalLM
from utils.utils import log_probs_from_logits
import multiprocessing as mp
import queue
import time
import os
import pickle
import copy
from typing import List, Tuple, Dict, Any, Optional
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VllmWorkerProcess:
    """
    独立的VllmSampler工作进程
    每个进程管理自己的GPU设备和VLLM实例
    """
    
    def __init__(self, worker_id: int, device_id: int, llm_kwargs: Dict, sae_kwargs: Dict, sampling_kwargs: Dict):
        self.worker_id = worker_id
        self.device_id = device_id
        self.llm_kwargs = llm_kwargs
        self.sae_kwargs = sae_kwargs
        self.sampling_kwargs = sampling_kwargs
        
        # 在进程中初始化时设置
        self.vllm_sampler = None
        self.initialized = False
        self.tokenizer = None
    
    def initialize(self):
        """在工作进程中初始化VllmSampler"""
        try:
            # 设置当前进程的CUDA设备
            if self.device_id == 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
            
            # 导入VllmSampler（在进程中导入避免主进程的依赖问题）
            from models.vllm_sampler import VllmSampler
            
            # 创建VllmSampler实例
            self.vllm_sampler = VllmSampler(
                self.llm_kwargs, 
                self.sae_kwargs, 
                self.sampling_kwargs, 
                device=str(self.device_id)
            )
            
            self.initialized = True
            logger.info(f"Worker {self.worker_id} initialized on GPU {self.device_id}")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id} initialization failed: {e}")
            self.initialized = False
            raise e
    
    def process_batch(self, batch_data: List[Tuple[str, List[float]]], samples_per_prompt: int = 1, progress_callback=None):
        """处理一个批次的数据"""
        if not self.initialized:
            raise RuntimeError(f"Worker {self.worker_id} not initialized")
        
        try:
            # 使用VllmSampler的generate_batch方法
            result = self.vllm_sampler.generate_batch(batch_data, samples_per_prompt, progress_callback=progress_callback)
            return result
        except Exception as e:
            logger.error(f"Worker {self.worker_id} batch processing failed: {e}")
            raise e
        
    def get_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = self.vllm_sampler.get_tokenizer()
            
        return self.tokenizer
    
    def get_model_info(self):
        """获取模型信息"""
        if not self.initialized:
            raise RuntimeError(f"Worker {self.worker_id} not initialized")
        if self.device_id == 1:
            return {
                'tokenizer': self.get_tokenizer(),
                'hidden_state_size': self.vllm_sampler.get_hidden_state_size(),
                'num_layers': self.vllm_sampler.get_num_layers()
            }
        else:
            return None
    
    def get_last_token_hidden_state(self, prompts: list[str], batch_size: int=8) -> torch.Tensor:
        """获取最后一个token的hidden state"""
        if not self.initialized:
            raise RuntimeError(f"Worker {self.worker_id} not initialized")
        
        try:
            result = self.vllm_sampler.get_last_token_hidden_state(prompts, batch_size)
            return result
        except Exception as e:
            logger.error(f"Worker {self.worker_id} get_last_token_hidden_state failed: {e}")
            raise e
    
    def get_logprobs(self, sequences: torch.Tensor, attention_masks: torch.Tensor, temperature: float=1.0, batch_size: int=4) -> torch.Tensor:
        """获取log probabilities"""
        if not self.initialized:
            raise RuntimeError(f"Worker {self.worker_id} not initialized")
        
        try:
            result = self.vllm_sampler.get_logprobs(sequences, attention_masks, temperature, batch_size)
            return result
        except Exception as e:
            logger.error(f"Worker {self.worker_id} get_logprobs failed: {e}")
            raise e


def worker_process_main(worker_id: int, device_id: int, llm_kwargs: Dict, sae_kwargs: Dict, 
                       sampling_kwargs: Dict, task_queue: mp.Queue, result_queue: mp.Queue, 
                       control_queue: mp.Queue):
    """
    工作进程的主函数
    
    Args:
        worker_id: 工作进程ID
        device_id: GPU设备ID
        llm_kwargs: LLM参数
        sae_kwargs: SAE参数
        sampling_kwargs: 采样参数
        task_queue: 任务队列
        result_queue: 结果队列
        control_queue: 控制队列
    """
    
    worker = None
    
    try:
        # 创建并初始化worker
        worker = VllmWorkerProcess(worker_id, device_id, llm_kwargs, sae_kwargs, sampling_kwargs)
        worker.initialize()
        
        # 通知主进程初始化完成
        control_queue.put(('initialized', worker_id, worker.get_model_info()))
        
        # 主工作循环
        while True:
            try:
                # 从任务队列获取任务
                task = task_queue.get(timeout=1.0)
                
                if task is None:  # 退出信号
                    logger.info(f"Worker {worker_id} received exit signal")
                    break
                
                # 解析任务类型
                if len(task) == 2:
                    task_id, method_name = task
                    if method_name == 'get_tokenizer':
                        result = worker.get_tokenizer()
                        result_queue.put((task_id, result, 0, None))

                elif len(task) == 3:
                    # 生成任务
                    task_id, batch_data, samples_per_prompt = task
                    
                    # 处理批次
                    start_time = time.time()
                    
                    # 创建进度条
                    total_samples = len(batch_data)
                    # with tqdm(total=total_samples, desc=f"Worker {worker_id} Processing", 
                    #          unit="sample", position=worker_id, leave=False) as pbar:
                    #     # 定义进度回调函数
                    #     def progress_callback(n):
                    #         pbar.update(n)
                    progress_callback=None
                    # 处理批次并传入回调函数
                    result = worker.process_batch(batch_data, samples_per_prompt, progress_callback=progress_callback)
                    
                    processing_time = time.time() - start_time
                    
                    # 发送结果
                    result_queue.put((task_id, result, processing_time, None))
                    
                elif len(task) == 4:
                    # 其他方法调用
                    task_id, method_name, args, kwargs = task
                    
                    start_time = time.time()
                    try:
                        if method_name == 'get_last_token_hidden_state':
                            result = worker.get_last_token_hidden_state(*args, **kwargs)
                        elif method_name == 'get_logprobs':
                            result = worker.get_logprobs(*args, **kwargs)
                        else:
                            raise ValueError(f"Unknown method: {method_name}")
                        
                        processing_time = time.time() - start_time
                        result_queue.put((task_id, result, processing_time, None))
                        
                    except Exception as e:
                        processing_time = time.time() - start_time
                        result_queue.put((task_id, None, processing_time, str(e)))
                else:
                    logger.error(f"Worker {worker_id} received invalid task format: {task}")
                    continue
                
            except queue.Empty:
                # 检查是否有控制信号
                try:
                    control_signal = control_queue.get_nowait()
                    if control_signal[0] == 'shutdown':
                        logger.info(f"Worker {worker_id} received shutdown signal")
                        break
                except queue.Empty:
                    continue
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} task processing error: {e}")
                # 发送错误结果
                result_queue.put((task_id, None, 0, str(e)))
    
    except Exception as e:
        logger.error(f"Worker {worker_id} fatal error: {e}")
        control_queue.put(('error', worker_id, str(e)))
    
    finally:
        logger.info(f"Worker {worker_id} shutting down")
        # 清理资源
        if worker and worker.initialized:
            try:
                # 这里可以添加清理代码
                pass
            except Exception as e:
                logger.error(f"Worker {worker_id} cleanup error: {e}")


class MultiProcessVllmSampler:
    """
    基于多进程的VllmSampler管理器
    
    特点：
    1. 每个进程独立管理GPU资源
    2. 避免Python GIL限制
    3. 更好的错误隔离
    4. 支持动态负载均衡
    """
    
    def __init__(self, llm_kwargs: Dict, sae_kwargs: Dict, sampling_kwargs: Dict, 
                 num_processes: int = 2, gpu_devices: Optional[List[int]] = None, 
                 max_queue_size: int = 100):
        """
        初始化多进程VllmSampler
        
        Args:
            llm_kwargs: LLM初始化参数
            sae_kwargs: SAE初始化参数
            sampling_kwargs: 采样参数
            num_processes: 进程数量
            gpu_devices: GPU设备列表，如果为None则自动分配
            max_queue_size: 最大队列大小
        """
        self.num_processes = num_processes
        self.max_queue_size = max_queue_size
        
        # 确定GPU设备分配
        if gpu_devices is None:
            available_gpus = list(range(torch.cuda.device_count()))
            if len(available_gpus) == 0:
                raise RuntimeError("No CUDA devices available")
            # 循环分配GPU
            self.gpu_devices = [available_gpus[i % len(available_gpus)] for i in range(num_processes)]
        else:
            if len(gpu_devices) < num_processes:
                # 如果GPU数量少于进程数量，循环分配
                self.gpu_devices = [gpu_devices[i % len(gpu_devices)] for i in range(num_processes)]
            else:
                self.gpu_devices = gpu_devices[:num_processes]
        
        logger.info(f"GPU device allocation: {self.gpu_devices}")
        
        # 复制参数以避免进程间共享问题
        self.llm_kwargs = copy.deepcopy(llm_kwargs)
        self.sae_kwargs = copy.deepcopy(sae_kwargs)
        self.sampling_kwargs = copy.deepcopy(sampling_kwargs)
        
        # 进程管理
        self.processes = []
        self.task_queues = []
        self.result_queue = None
        self.control_queues = []
        
        # 状态管理
        self.initialized = False
        self.model_info = None
        self.task_counter = 0
        
        # 性能统计
        self.total_tasks = 0
        self.total_processing_time = 0.0
        self.process_usage_count = [0] * num_processes
        
        # 初始化进程
        self._initialize_processes()
    
    def _check_process_status(self):
        """检查所有进程的状态（调试用）"""
        logger.info("=== Process Status Check ===")
        for i, process in enumerate(self.processes):
            logger.info(f"Process {i} (PID: {process.pid if process.pid else 'None'}): "
                       f"alive={process.is_alive()}, GPU={self.gpu_devices[i]}")
        
        logger.info("=== Queue Status Check ===")
        for i, control_queue in enumerate(self.control_queues):
            try:
                queue_size = control_queue.qsize()
                logger.info(f"Control queue {i} size: {queue_size}")
            except:
                logger.info(f"Control queue {i} size: unknown")
        
        logger.info("=== End Status Check ===")
        return True
    
    def _initialize_processes(self):
        """初始化所有工作进程"""
        logger.info(f"Initializing {self.num_processes} worker processes...")
        
        # 创建队列
        self.result_queue = mp.Queue(maxsize=self.max_queue_size)
        
        for i in range(self.num_processes):
            # 为每个进程创建独立的任务队列和控制队列
            task_queue = mp.Queue(maxsize=self.max_queue_size)
            control_queue = mp.Queue()
            
            self.task_queues.append(task_queue)
            self.control_queues.append(control_queue)
            
            # 创建进程
            process = mp.Process(
                target=worker_process_main,
                args=(i, self.gpu_devices[i], self.llm_kwargs, self.sae_kwargs, 
                     self.sampling_kwargs, task_queue, self.result_queue, control_queue)
            )
            
            process.start()
            self.processes.append(process)
            logger.info(f"Started worker process {i} (PID: {process.pid}) on GPU {self.gpu_devices[i]}")
            
            # 在启动下一个进程前等待1秒，避免资源竞争
            if i < self.num_processes - 1:  # 最后一个进程不需要等待
                time.sleep(1.0)
                logger.info(f"Waiting 1 second before starting next process...")
        
        # 等待所有进程初始化完成
        self._wait_for_initialization()
    
    def _wait_for_initialization(self, timeout: float = 60.0):
        """等待所有进程初始化完成"""
        initialized_count = 0
        initialized_workers = set()
        start_time = time.time()
        
        logger.info(f"Waiting for {self.num_processes} worker processes to initialize...")
        last_status_check = start_time
        
        while initialized_count < self.num_processes:
            # if time.time() - start_time > timeout:
            #     alive_processes = [p.is_alive() for p in self.processes]
            #     logger.error(f"Process initialization timeout after {timeout} seconds")
            #     logger.error(f"Initialized: {initialized_count}/{self.num_processes}")
            #     logger.error(f"Process alive status: {alive_processes}")
            #     logger.error(f"Initialized workers: {initialized_workers}")
            #     raise TimeoutError(f"Process initialization timeout after {timeout} seconds. "
            #                      f"Only {initialized_count}/{self.num_processes} processes initialized.")
            
            # 检查控制队列中的初始化消息
            for i, control_queue in enumerate(self.control_queues):
                try:
                    message = control_queue.get_nowait()
                    if message[0] == 'initialized':
                        worker_id = message[1]
                        model_info = message[2]
                        
                        # 避免重复计数
                        if worker_id not in initialized_workers:
                            initialized_workers.add(worker_id)
                            initialized_count += 1
                            
                            # 只从第一个worker获取模型信息
                            if self.model_info is None and model_info is not None:
                                self.model_info = model_info
                                logger.info(f"Model info received from worker {worker_id}")
                        
                            logger.info(f"Worker {worker_id} initialized successfully ({initialized_count}/{self.num_processes})")
                        
                    elif message[0] == 'error':
                        worker_id = message[1]
                        error_msg = message[2]
                        logger.error(f"Worker {worker_id} initialization error: {error_msg}")
                        raise RuntimeError(f"Worker {worker_id} initialization failed: {error_msg}")
                        
                except queue.Empty:
                    continue
            
            # 检查进程是否还活着
            for i, process in enumerate(self.processes):
                if not process.is_alive() and i not in initialized_workers:
                    logger.error(f"Process {i} died before initialization")
                    raise RuntimeError(f"Process {i} died before initialization")
            
            # 定期打印状态检查（每10秒）
            current_time = time.time()
            if current_time - last_status_check > 10.0:
                logger.info(f"Still waiting... ({initialized_count}/{self.num_processes} initialized, "
                           f"{current_time - start_time:.1f}s elapsed)")
                self._check_process_status()
                last_status_check = current_time
            
            time.sleep(0.1)
        
        # 确保我们有模型信息
        if self.model_info is None:
            logger.warning("No model info received, this might cause issues later")
        
        self.initialized = True
        logger.info(f"All {self.num_processes} worker processes initialized successfully")
    
    def _split_data(self, prompts: List[str], strengths_list: List[List[float]]) -> List[List[Tuple[str, List[float]]]]:
        """将数据分割成多个批次，分配给不同的进程"""
        data = list(zip(prompts, strengths_list))
        batch_size = len(data) // self.num_processes
        remainder = len(data) % self.num_processes
        
        batches = []
        start_idx = 0
        
        for i in range(self.num_processes):
            # 为前remainder个批次多分配一个样本
            current_batch_size = batch_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_batch_size
            
            if start_idx < len(data):
                batches.append(data[start_idx:end_idx])
            else:
                batches.append([])
            
            start_idx = end_idx
        
        return batches
    
    def generate_parallel(self, prompts: List[str], strengths_list: List[List[float]], 
                         samples_per_prompt: int = 1, timeout: float = 300.0, batch_size: int = 1):
        """
        并行生成文本 - 平均分配版本
        
        Args:
            prompts: 输入prompt列表
            strengths_list: 对应的strengths列表
            samples_per_prompt: 每个prompt生成的样本数
            timeout: 超时时间（秒）
            batch_size: 每个任务的批次大小（在平均分配模式下此参数被忽略）
            
        Returns:
            所有生成的输出列表
        """
        if not self.initialized:
            raise RuntimeError("MultiProcessVllmSampler not initialized")
        
        if len(prompts) == 0:
            return []
        
        # 使用平均分配策略将数据分配给各个进程
        process_batches = self._split_data(prompts, strengths_list)
        
        # 为每个进程分配任务
        task_ids = []
        process_task_mapping = {}  # 记录进程对应的任务ID
        
        start_time = time.time()
        
        logger.info(f"Starting even distribution for {len(prompts)} prompts across {self.num_processes} processes")
        
        # 分配任务给各个进程
        for process_id, batch_data in enumerate(process_batches):
            if len(batch_data) > 0:  # 只为有数据的进程分配任务
                task_id = self.task_counter
                self.task_counter += 1
                task_ids.append(task_id)
                process_task_mapping[task_id] = process_id
                
                try:
                    # 将任务放入对应进程的队列
                    self.task_queues[process_id].put((task_id, batch_data, samples_per_prompt), timeout=timeout)
                    self.process_usage_count[process_id] += 1
                    
                    logger.info(f"Assigned {len(batch_data)} prompts to process {process_id} (task {task_id})")
                    
                except queue.Full:
                    logger.error(f"Process {process_id} queue is full, skipping task {task_id}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to assign task {task_id} to process {process_id}: {e}")
                    continue
        
        # 收集结果
        results = {}
        completed_tasks = 0
        expected_tasks = len(task_ids)
        
        logger.info(f"Waiting for {expected_tasks} tasks to complete")
        
        while completed_tasks < expected_tasks:
            try:
                task_id, result, processing_time, error = self.result_queue.get(timeout=timeout)
                
                if error is not None:
                    logger.error(f"Task {task_id} failed: {error}")
                    results[task_id] = []
                else:
                    results[task_id] = result
                    self.total_processing_time += processing_time
                    
                    if task_id in process_task_mapping:
                        process_id = process_task_mapping[task_id]
                        logger.debug(f"Process {process_id} completed task {task_id} with {len(result)} outputs")
                
                completed_tasks += 1
                
                # 打印进度
                if completed_tasks % max(1, expected_tasks // 10) == 0 or completed_tasks == expected_tasks:
                    logger.info(f"Progress: {completed_tasks}/{expected_tasks} tasks completed")
                    
            except queue.Empty:
                logger.error(f"Timeout waiting for task results after {timeout} seconds")
                logger.error(f"Completed: {completed_tasks}/{expected_tasks}")
                break
            except Exception as e:
                logger.error(f"Error while collecting results: {e}")
                break
        
        # 按进程顺序合并结果，保持原始顺序
        all_outputs = []
        for task_id in task_ids:
            if task_id in results:
                all_outputs.extend(results[task_id])
            else:
                logger.warning(f"Task {task_id} result missing")
        
        self.total_tasks += len(prompts)
        
        # 打印分配统计
        distribution_stats = {}
        for process_id, batch_data in enumerate(process_batches):
            distribution_stats[process_id] = len(batch_data)
        
        logger.info(f"Even distribution stats: {distribution_stats}")
        logger.info(f"Total processing time: {self.total_processing_time:.2f}s")
        
        return all_outputs
    
    def generate(self, prompts: List[str], strengths_list: List[List[float]], 
                samples_per_prompt: int = 1, use_parallel: bool = True, timeout: float = 300.0, batch_size: int = 1):
        """
        生成文本（统一接口）
        
        Args:
            prompts: 输入prompt列表
            strengths_list: 对应的strengths列表
            samples_per_prompt: 每个prompt生成的样本数
            use_parallel: 是否使用并行（对于多进程版本，这个参数总是True）
            timeout: 超时时间
            batch_size: 每个任务的批次大小
            
        Returns:
            生成的输出列表
        """
        return self.generate_parallel(prompts, strengths_list, samples_per_prompt, timeout, batch_size)
    
    def get_tokenizer(self, device=4):
        """获取tokenizer"""
        # if not self.initialized or self.model_info is None:
        #     raise RuntimeError("MultiProcessVllmSampler not initialized")
        # task_id = f"tokenizer_{time.time()}"
        # task = (task_id, 'get_tokenizer')
        # self.task_queues[device-1].put(task, timeout=10.0)
        # # 等待结果
        # start_time = time.time()
        # while True:
        #     try:
        #         result_task_id, result, processing_time, error = self.result_queue.get(timeout=1.0)
        #         if result_task_id == task_id:
        #             if error:
        #                 raise RuntimeError(f"Worker process error: {error}")
        #             return result
        #     except queue.Empty:
        #         continue
        if not self.initialized or self.model_info is None:
            raise RuntimeError("MultiProcessVllmSampler not initialized")
        return self.model_info['tokenizer']
    
    def get_hidden_state_size(self) -> int:
        """获取隐藏状态大小"""
        if not self.initialized or self.model_info is None:
            raise RuntimeError("MultiProcessVllmSampler not initialized")
        return self.model_info['hidden_state_size']
    
    def get_num_layers(self) -> int:
        """获取层数"""
        if not self.initialized or self.model_info is None:
            raise RuntimeError("MultiProcessVllmSampler not initialized")
        return self.model_info['num_layers']

    def get_last_token_hidden_state(self, prompts: list[str], batch_size: int=8)->torch.Tensor:
        """获取最后一个token的hidden state，使用第一个进程处理"""
        if not self.initialized:
            raise RuntimeError("MultiProcessVllmSampler not initialized")
        
        # 使用第一个进程处理
        task_id = f"hidden_state_{time.time()}"
        task = (task_id, 'get_last_token_hidden_state', (prompts, batch_size), {})
        
        try:
            self.task_queues[0].put(task, timeout=10.0)
            
            # 等待结果
            start_time = time.time()
            # while time.time() - start_time < 300.0:  # 5分钟超时
            while True:
                try:
                    result_task_id, result, processing_time, error = self.result_queue.get(timeout=1.0)
                    if result_task_id == task_id:
                        if error:
                            raise RuntimeError(f"Worker process error: {error}")
                        return result
                except queue.Empty:
                    continue
            
        except Exception as e:
            logger.error(f"get_last_token_hidden_state failed: {e}")
            raise e

    def get_logprobs(self, sequences: torch.Tensor, attention_masks: torch.Tensor, temperature: float=1.0, batch_size: int=4)->torch.Tensor:
        """获取log probabilities，使用第一个进程处理"""
        if not self.initialized:
            raise RuntimeError("MultiProcessVllmSampler not initialized")
        
        # 使用第一个进程处理
        task_id = f"logprobs_{time.time()}"
        task = (task_id, 'get_logprobs', (sequences, attention_masks, temperature, batch_size), {})
        
        try:
            self.task_queues[0].put(task, timeout=10.0)
            
            # 等待结果
            start_time = time.time()
            # while time.time() - start_time < 300.0:  # 5分钟超时
            while True:
                try:
                    result_task_id, result, processing_time, error = self.result_queue.get(timeout=1.0)
                    if result_task_id == task_id:
                        if error:
                            raise RuntimeError(f"Worker process error: {error}")
                        return result
                except queue.Empty:
                    continue
            
            raise TimeoutError("get_logprobs timeout")
            
        except Exception as e:
            logger.error(f"get_logprobs failed: {e}")
            raise e
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if not self.initialized:
            raise RuntimeError("MultiProcessVllmSampler not initialized")
        
        if self.total_tasks == 0:
            return {
                'total_tasks': 0,
                'total_processing_time': 0.0,
                'avg_processing_time': 0.0,
                'process_usage_count': self.process_usage_count.copy(),
                'gpu_devices': self.gpu_devices.copy(),
                'load_balance_ratio': [0.0] * self.num_processes,
                'load_variance': 0.0,
                'num_processes': self.num_processes
            }
        
        avg_processing_time = self.total_processing_time / self.total_tasks
        
        # 计算负载均衡指标
        total_usage = sum(self.process_usage_count)
        load_balance_ratio = [count / max(total_usage, 1) for count in self.process_usage_count]
        ideal_ratio = 1.0 / self.num_processes
        load_variance = sum((ratio - ideal_ratio)**2 for ratio in load_balance_ratio) / self.num_processes
        
        return {
            'total_tasks': self.total_tasks,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': avg_processing_time,
            'process_usage_count': self.process_usage_count.copy(),
            'gpu_devices': self.gpu_devices.copy(),
            'process_usage_ratio': load_balance_ratio,
            'load_balance_ratio': load_balance_ratio,
            'load_variance': load_variance,  # 越小表示负载越均衡
            'num_processes': self.num_processes
        }
    
    def shutdown(self, timeout: float = 30.0):
        """关闭所有工作进程"""
        logger.info("Shutting down MultiProcessVllmSampler...")
        
        # 发送退出信号
        for i, task_queue in enumerate(self.task_queues):
            try:
                task_queue.put(None, timeout=5.0)  # None是退出信号
            except queue.Full:
                logger.warning(f"Could not send exit signal to process {i}")
        
        # 等待进程结束
        for i, process in enumerate(self.processes):
            try:
                process.join(timeout=timeout)
                if process.is_alive():
                    logger.warning(f"Force terminating process {i}")
                    process.terminate()
                    process.join(timeout=5.0)
                    if process.is_alive():
                        logger.error(f"Could not terminate process {i}")
                        process.kill()
            except Exception as e:
                logger.error(f"Error shutting down process {i}: {e}")
        
        # 清理队列
        try:
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except:
            pass
        
        for task_queue in self.task_queues:
            try:
                while not task_queue.empty():
                    task_queue.get_nowait()
            except:
                pass
        
        logger.info("MultiProcessVllmSampler shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def __del__(self):
        if hasattr(self, 'processes') and self.processes:
            self.shutdown()


# 为了向后兼容，提供一个工厂函数
def create_multiprocess_vllm_sampler(llm_kwargs: Dict, sae_kwargs: Dict, sampling_kwargs: Dict, 
                                    num_processes: int = None, gpu_devices: List[int] = None) -> MultiProcessVllmSampler:
    """
    创建多进程VllmSampler的工厂函数
    
    Args:
        llm_kwargs: LLM参数
        sae_kwargs: SAE参数
        sampling_kwargs: 采样参数
        num_processes: 进程数量，如果为None则根据GPU数量自动确定
        gpu_devices: GPU设备列表
        
    Returns:
        MultiProcessVllmSampler实例
    """
    if num_processes is None:
        # 根据GPU数量自动确定进程数
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise RuntimeError("No CUDA devices available")
        elif gpu_count == 1:
            num_processes = 1
        elif gpu_count == 2:
            num_processes = 2
        else:
            num_processes = min(gpu_count, 4)  # 最多4个进程
    
    return MultiProcessVllmSampler(
        llm_kwargs=llm_kwargs,
        sae_kwargs=sae_kwargs,
        sampling_kwargs=sampling_kwargs,
        num_processes=num_processes,
        gpu_devices=gpu_devices
    )