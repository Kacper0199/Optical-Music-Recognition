import threading
import time
import psutil
import torch
import os
import matplotlib.pyplot as plt


class PerformanceMonitor:
    def __init__(self):
        self.running = False
        self.thread = None
        self.start_time = 0

        self.time_points = []
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_mem_usage = []

        self.request_indices = []
        self.latencies = []
        self.throughputs = []

        self.device_type = "cpu"
        if torch.cuda.is_available():
            self.device_type = "cuda"
        elif torch.backends.mps.is_available():
            self.device_type = "mps"

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self, output_dir):
        self.running = False
        if self.thread:
            self.thread.join()
        self._save_plots(output_dir)

    def log_inference(self, latency, num_chars):
        if latency <= 0:
            return

        est_tokens = num_chars / 4.0
        tps = est_tokens / latency

        self.request_indices.append(len(self.request_indices) + 1)
        self.latencies.append(latency)
        self.throughputs.append(tps)

    def _monitor_loop(self):
        process = psutil.Process(os.getpid())

        while self.running:
            current_t = time.time() - self.start_time

            cpu = psutil.cpu_percent()
            ram = process.memory_info().rss / (1024 * 1024)

            gpu = 0.0
            try:
                if self.device_type == "cuda":
                    gpu = torch.cuda.memory_allocated() / (1024 * 1024)
                elif self.device_type == "mps":
                    gpu = torch.mps.current_allocated_memory() / (1024 * 1024)
            except:
                pass

            self.time_points.append(current_t)
            self.cpu_usage.append(cpu)
            self.ram_usage.append(ram)
            self.gpu_mem_usage.append(gpu)

            time.sleep(0.5)

    def _save_plots(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.cpu_usage, label='CPU Usage')
        plt.xlabel('Time [s]')
        plt.ylabel('CPU [%]')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'cpu_usage.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.ram_usage, label='RAM Usage', color='orange')
        plt.xlabel('Time [s]')
        plt.ylabel('RAM [MB]')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'ram_usage.png'))
        plt.close()

        if any(v > 0 for v in self.gpu_mem_usage):
            plt.figure(figsize=(10, 6))
            plt.plot(self.time_points, self.gpu_mem_usage, label='GPU Memory', color='green')
            plt.xlabel('Time [s]')
            plt.ylabel('GPU Mem [MB]')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'gpu_memory.png'))
            plt.close()

        if self.request_indices:
            plt.figure(figsize=(10, 6))
            plt.plot(self.request_indices, self.latencies, marker='o', color='red')
            plt.xlabel('Image Index')
            plt.ylabel('Latency [s]')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'latency.png'))
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.bar(self.request_indices, self.throughputs, color='purple')
            plt.xlabel('Image Index')
            plt.ylabel('Throughput [tokens/s]')
            plt.grid(axis='y')
            plt.savefig(os.path.join(output_dir, 'throughput.png'))
            plt.close()
