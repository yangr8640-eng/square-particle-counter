"""Launch with: python -m local_app [--model /path/to/particle_filter.joblib]."""
from __future__ import annotations
import argparse
from pathlib import Path
import threading
import webbrowser

try:
    from waitress import create_server
    from .server import create_app
    from .pipeline import prepare_model
except (ImportError, OSError) as exc:
    raise SystemExit(f'无法启动：运行依赖缺失或无法载入，请重新解压完整程序包。\n具体原因：{exc}') from None


def main():
    parser = argparse.ArgumentParser(description='离线颗粒计数：本地上传、复核和结果下载。')
    parser.add_argument('--model', type=Path, help='指定本地模型，可替换为新版本')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--port', type=int, default=0, help='本机端口；默认随机分配')
    parser.add_argument('--data-dir', type=Path, help='专用任务保存目录；默认使用退出时清理的临时目录')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('端口必须在 0 到 65535 之间。')
    try:
        app = create_app(args.data_dir, model_path=args.model)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'无法启动：{exc}\n')
    manager = app.extensions['particle_jobs']
    try:
        prepare_model(manager.model_path)
    except Exception as exc:
        manager.close()
        parser.exit(1, f'无法启动：本地模型或运行依赖未通过检查，请确认程序文件完整。\n具体原因：{exc}\n')
    try:
        server = create_server(app, host='127.0.0.1', port=args.port, threads=4,
                               max_request_body_size=app.config['MAX_CONTENT_LENGTH'])
    except OSError as exc:
        manager.close()
        parser.exit(1, f'无法打开本机端口：{exc}\n')
    url = f'http://127.0.0.1:{server.effective_port}/'
    print(f'本地颗粒计数已启动：{url}', flush=True)
    print('图片仅在本机处理。请保持此窗口打开，退出时按 Ctrl+C。', flush=True)
    if not args.no_browser:
        timer = threading.Timer(.3, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        manager.close()


if __name__ == '__main__':
    main()
