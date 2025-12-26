#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投資分析ツールキット API パッケージのエントリーポイント

使用方法:
    python -m investment_toolkit.api [--port PORT]

これにより watchlist_api サーバーが起動します。
"""

if __name__ == "__main__":
    from .watchlist_api import run_server
    import argparse

    parser = argparse.ArgumentParser(description='ウォッチリスト API サーバー')
    parser.add_argument('--host', default='127.0.0.1', help='ホスト (デフォルト: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5001, help='ポート (デフォルト: 5001)')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug)
