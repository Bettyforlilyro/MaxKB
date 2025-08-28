# coding=utf-8
"""
    @project: maxkb
    @Author：虎
    @file： embedding_config.py
    @date：2023/10/23 16:03
    @desc:
"""
import threading
import time

from common.cache.mem_cache import MemCache

_lock = threading.Lock()
locks = {}


class ModelManage:
    cache = MemCache('model', {})
    up_clear_time = time.time()

    @staticmethod
    def _get_lock(_id):
        lock = locks.get(_id)
        if lock is None:
            with _lock:
                lock = locks.get(_id)
                if lock is None:
                    lock = threading.Lock()
                    locks[_id] = lock

        return lock

    @staticmethod
    def get_model(_id, get_model):
        # 优先尝试从缓存中获取已经加载好的模型
        model_instance = ModelManage.cache.get(_id)
        if model_instance is None:  # 如果缓存中没有，则获取锁再尝试加载模型
            lock = ModelManage._get_lock(_id)
            with lock:
                # 这里为什么要在加了锁之后第二次检查缓存？
                # ---因为如果不检查，前面可能会存在多个线程同时认为“缓存中没有加载好的模型”，都在等待锁，如果这里不检查，就会再次加载模型，导致重复加载显存爆炸
                model_instance = ModelManage.cache.get(_id)     # 再次尝试去缓存中查询是否已经存在加载好的模型
                if model_instance is None:
                    model_instance = get_model(_id)     # 缓存中没有模型实例，真正地去加载模型并保存到缓存中，设置超时自动清理时间为8h
                    ModelManage.cache.set(_id, model_instance, timeout=60 * 60 * 8)
        else:
            if model_instance.is_cache_model():
                ModelManage.cache.touch(_id, timeout=60 * 60 * 8)
            else:   # 不能被缓存的模型实例（通常是API类模型即通过API URL + API KEY调用的LLM。各种配置可能是动态的，比如API KEY可能会刷新的情况）
                model_instance = get_model(_id)
                ModelManage.cache.set(_id, model_instance, timeout=60 * 60 * 8)
        ModelManage.clear_timeout_cache()
        return model_instance

    @staticmethod
    def clear_timeout_cache():
        # 每小时启动一个异步线程清理过期的缓存，不阻塞主进程
        if time.time() - ModelManage.up_clear_time > 60 * 60:
            threading.Thread(target=lambda: ModelManage.cache.clear_timeout_data()).start()
            ModelManage.up_clear_time = time.time()

    @staticmethod
    def delete_key(_id):
        if ModelManage.cache.has_key(_id):
            ModelManage.cache.delete(_id)


class VectorStore:
    from embedding.vector.pg_vector import PGVector
    from embedding.vector.base_vector import BaseVectorStore
    instance_map = {
        'pg_vector': PGVector,
    }
    instance = None

    @staticmethod
    def get_embedding_vector() -> BaseVectorStore:
        from embedding.vector.pg_vector import PGVector
        if VectorStore.instance is None:
            from smartdoc.const import CONFIG
            vector_store_class = VectorStore.instance_map.get(CONFIG.get("VECTOR_STORE_NAME"),
                                                              PGVector)
            VectorStore.instance = vector_store_class()
        return VectorStore.instance
