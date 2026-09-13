import argparse
import sys

from ..config import StreamTrainConfig
from ..logging.logger import logger
from ..resources.monitor import ResourceMonitor


def cmd_system_info(args):
    monitor = ResourceMonitor()
    info = monitor.get_system_info()
    print("StreamTrain System Information:")
    print("-------------------------------")
    for k, v in info.items():
        print(f"{k.replace('_', ' ').title()}: {v}")

def cmd_train(args):
    try:
        config = StreamTrainConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
        
    logger.info("Initializing Trainer from config...")
    logger.info("Note: The CLI 'train' command usually expects an entrypoint script where your model and dataset are defined.")
    logger.info("For a complete example, run 'python examples/synthetic_train.py'")
    # Real framework would likely invoke a user-provided script here or expect the user to use the Trainer programmatically.

def cmd_resume(args):
    logger.info("Resuming training...")
    logger.info("For a complete example, run 'python examples/synthetic_train.py --resume'")

def cmd_status(args):
    logger.info("Status: Not currently tracking a background training process.")
    
def cmd_monitor(args):
    logger.info("Monitoring system resources (Ctrl+C to stop)...")
    monitor = ResourceMonitor()
    try:
        while True:
            info = monitor.get_system_info()
            ram_gb = monitor.get_current_ram_usage_gb()
            vram_gb = monitor.get_current_vram_usage_gb()
            cpu_percent = monitor.check_cpu_pressure() # Just a boolean placeholder, could get actual %
            import psutil
            cpu_val = psutil.cpu_percent(interval=1)
            print(f"RAM: {ram_gb:.2f}GB / {info.get('ram_total_gb', 0):.2f}GB | CPU: {cpu_val}% | VRAM: {vram_gb:.2f}GB")
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def main():
    parser = argparse.ArgumentParser(description="StreamTrain CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parser_train = subparsers.add_parser("train", help="Start training from a configuration file")
    parser_train.add_argument("config", type=str, help="Path to YAML configuration")
    parser_train.add_argument("--mode", type=str, default="balanced", help="Training mode (performance, balanced, laptop)")
    
    parser_resume = subparsers.add_parser("resume", help="Resume training from latest checkpoint")
    parser_pause = subparsers.add_parser("pause", help="Pause a background training process (placeholder)")
    parser_stop = subparsers.add_parser("stop", help="Stop a background training process (placeholder)")
    parser_status = subparsers.add_parser("status", help="Show status of training process")
    parser_monitor = subparsers.add_parser("monitor", help="Monitor system resources")
    parser_sysinfo = subparsers.add_parser("system-info", help="Show system hardware and environment info")

    args = parser.parse_args()

    if args.command == "system-info":
        cmd_system_info(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "resume":
        cmd_resume(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command in ["pause", "stop"]:
        logger.info(f"'{args.command}' signal sent (this is a placeholder in the MVP).")

if __name__ == "__main__":
    main()
