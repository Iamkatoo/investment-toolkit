#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ウォッチリスト機能用の簡単なAPIサーバー
将来的なWebアプリケーション統合のための基盤
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# プロジェクトルート（investment-toolkitのルート）をパスに追加
# watchlist_api.py -> api -> investment_toolkit -> src -> investment-toolkit
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask is not available. API server will run in simulation mode.")
    FLASK_AVAILABLE = False

from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from investment_toolkit.analysis.watchlist_manager import WatchlistManager, format_watchlist_metadata
from sqlalchemy import create_engine, text


# Flask アプリケーション設定
if FLASK_AVAILABLE:
    app = Flask(__name__)
    CORS(app)
    
    # データベース接続
    def get_engine():
        """データベースエンジンを取得"""
        try:
            SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            return create_engine(SQLALCHEMY_DATABASE_URI)
        except Exception as e:
            print(f"データベース接続エラー: {e}")
            return None
    
    @app.route('/api/watchlist/add', methods=['POST'])
    def add_to_watchlist():
        """ウォッチリストに銘柄を追加"""
        try:
            engine = get_engine()
            if not engine:
                return jsonify({'error': 'データベース接続エラー'}), 500
            
            wm = WatchlistManager(engine)
            stocks_data = request.get_json()
            
            if not stocks_data:
                return jsonify({'error': '銘柄データが必要です'}), 400
            
            # データ形式の変換
            formatted_stocks = []
            for stock in stocks_data:
                formatted_stock = {
                    'symbol': stock['symbol'],
                    'analysis_type': stock['analysisType'],
                    'metadata': stock['metadata'],
                    'analysis_category': stock['metadata'].get('analysis_category'),
                    'added_reason': f"ウォッチリスト機能から追加 - {stock['analysisType']}",
                    'notes': None
                }
                formatted_stocks.append(formatted_stock)
            
            result = wm.add_multiple_stocks(formatted_stocks)
            
            return jsonify({
                'success': True,
                'message': f'{result["success_count"]}銘柄をウォッチリストに追加しました',
                'success_count': result['success_count'],
                'failure_count': result['failure_count'],
                'total_count': result['total_count'],
                'errors': result['errors']
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/watchlist', methods=['GET'])
    def get_watchlist():
        """現在のウォッチリストを取得"""
        try:
            engine = get_engine()
            if not engine:
                return jsonify({'error': 'データベース接続エラー'}), 500
            
            wm = WatchlistManager(engine)
            analysis_type = request.args.get('analysis_type')
            lightweight = request.args.get('lightweight', 'false').lower() == 'true'
            
            if lightweight:
                # 軽量版：基本情報のみ取得
                watchlist_data = wm.get_lightweight_watchlist(analysis_type)
            else:
                # 通常版：詳細な情報を取得
                watchlist_data = wm.get_current_watchlist(analysis_type)
            
            return jsonify({
                'success': True,
                'data': watchlist_data.to_dict('records'),
                'count': len(watchlist_data)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/watchlist/performance', methods=['GET'])
    def get_performance_summary():
        """ウォッチリストのパフォーマンスサマリーを取得"""
        try:
            engine = get_engine()
            if not engine:
                return jsonify({'error': 'データベース接続エラー'}), 500
            
            wm = WatchlistManager(engine)
            analysis_type = request.args.get('analysis_type')
            days_back = int(request.args.get('days_back', 30))
            
            performance = wm.get_performance_summary(analysis_type, days_back)
            
            return jsonify({
                'success': True,
                'performance': performance
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/watchlist/stock/<symbol>', methods=['GET'])
    def get_stock_performance(symbol):
        """特定銘柄のパフォーマンス履歴を取得"""
        try:
            engine = get_engine()
            if not engine:
                return jsonify({'error': 'データベース接続エラー'}), 500
            
            wm = WatchlistManager(engine)
            analysis_type = request.args.get('analysis_type')
            
            performance_history = wm.get_stock_performance_history(symbol, analysis_type)
            
            return jsonify({
                'success': True,
                'symbol': symbol,
                'data': performance_history.to_dict('records'),
                'count': len(performance_history)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/watchlist/remove/<int:stock_id>', methods=['DELETE'])
    def remove_from_watchlist(stock_id):
        """ウォッチリストから銘柄を削除"""
        try:
            engine = get_engine()
            if not engine:
                return jsonify({'error': 'データベース接続エラー'}), 500
            
            wm = WatchlistManager(engine)
            data = request.get_json() or {}
            removal_reason = data.get('reason', 'ユーザーによる削除')
            
            success = wm.remove_stock_from_watchlist(stock_id, removal_reason)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'銘柄ID {stock_id} をウォッチリストから削除しました'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'銘柄ID {stock_id} の削除に失敗しました'
                }), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/watchlist/remove', methods=['POST'])
    def remove_by_symbol():
        """シンボルと分析タイプでウォッチリストから銘柄を削除"""
        try:
            engine = get_engine()
            if not engine:
                return jsonify({'error': 'データベース接続エラー'}), 500
            
            data = request.get_json()
            if not data or 'symbol' not in data or 'analysis_type' not in data:
                return jsonify({'error': 'symbol と analysis_type が必要です'}), 400
            
            symbol = data['symbol']
            analysis_type = data['analysis_type']
            removal_reason = data.get('reason', 'ユーザーによる削除')
            
            # シンボルと分析タイプでウォッチリスト項目を論理削除（is_active = false）
            remove_query = text("""
            UPDATE watchlist.tracked_stocks 
            SET is_active = false, 
                removed_date = CURRENT_TIMESTAMP,
                removal_reason = :reason
            WHERE symbol = :symbol AND analysis_type = :analysis_type AND is_active = true
            """)
            
            with engine.connect() as conn:
                result = conn.execute(remove_query, {
                    'symbol': symbol,
                    'analysis_type': analysis_type,
                    'reason': removal_reason
                })
                conn.commit()
                
                if result.rowcount > 0:
                    print(f"銘柄 {symbol} をウォッチリストから削除しました")
                    return jsonify({
                        'success': True,
                        'message': f'{symbol} をウォッチリストから削除しました'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': f'{symbol} がウォッチリストに見つかりません'
                    }), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/generate_detailed_report', methods=['POST'])
    def generate_detailed_report():
        """詳細レポート生成"""
        try:
            data = request.get_json()
            if not data or 'symbol' not in data:
                return jsonify({
                    'success': False,
                    'error': 'symbolパラメータが必要です'
                }), 400
            
            symbol = data['symbol'].strip()
            if not symbol:
                return jsonify({
                    'success': False,
                    'error': '有効なsymbolを指定してください'
                }), 400
            
            print(f"🚀 詳細レポート生成開始: {symbol}")
            
            # generate_single_stock_report.pyをコマンドライン引数付きで実行
            import subprocess
            import sys
            from pathlib import Path
            
            # スクリプトのパス（環境変数から取得、デフォルトは investment-workspace）
            workspace_root = os.getenv(
                'INVESTMENT_WORKSPACE_ROOT',
                str(Path(project_root).parent / 'investment-workspace')
            )
            script_path = Path(workspace_root) / "scripts" / "generate_single_stock_report.py"

            if not script_path.exists():
                return jsonify({
                    'success': False,
                    'error': f'generate_single_stock_report.py が見つかりません: {script_path}'
                }), 500

            # サブプロセスでスクリプトを実行
            cmd = [sys.executable, str(script_path), '--symbol', symbol, '--no-browser']

            print(f"📝 実行コマンド: {' '.join(cmd)}")

            # タイムアウト付きで実行（60秒）
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(workspace_root)
            )
            
            if result.returncode == 0:
                print(f"✅ {symbol} の詳細レポート生成成功")

                # 生成されたレポートファイルのパスを推定
                # レポートは REPORTS_BASE_DIR/individual_stocks/{symbol}_{company_name}/ に保存される
                from investment_toolkit.utilities.paths import get_or_create_reports_config
                reports_config = get_or_create_reports_config()
                reports_base_dir = reports_config.base_dir
                individual_stocks_dir = reports_config.individual_stocks_dir

                # 最新の銘柄ディレクトリを探す
                symbol_dirs = [d for d in individual_stocks_dir.glob(f"{symbol}_*") if d.is_dir()]

                if symbol_dirs:
                    # 最新のディレクトリを取得
                    latest_dir = max(symbol_dirs, key=lambda x: x.stat().st_mtime)

                    # ディレクトリ内の最新のHTMLファイルを探す
                    html_files = list(latest_dir.glob("*.html"))
                    if html_files:
                        latest_html = max(html_files, key=lambda x: x.stat().st_mtime)

                        # reports ディレクトリからの相対パスを計算
                        relative_path = latest_html.relative_to(reports_base_dir)

                        # HTTP URL を生成（APIサーバー経由でファイル配信）
                        report_url = f"http://127.0.0.1:5001/api/reports/{relative_path}"

                        return jsonify({
                            'success': True,
                            'message': f'{symbol} の詳細レポートを生成しました',
                            'report_url': report_url,
                            'report_path': str(latest_html),
                            'file_url': f"file://{latest_html.absolute()}"  # 念のためfile URLも残す
                        })
                
                # ファイルが見つからない場合でも成功とする
                return jsonify({
                    'success': True,
                    'message': f'{symbol} の詳細レポートを生成しました（ファイルパスの特定に失敗）',
                    'stdout': result.stdout
                })
            else:
                print(f"❌ {symbol} の詳細レポート生成失敗")
                print(f"標準出力: {result.stdout}")
                print(f"標準エラー: {result.stderr}")
                
                return jsonify({
                    'success': False,
                    'error': f'レポート生成に失敗しました',
                    'details': result.stderr or result.stdout,
                    'return_code': result.returncode
                }), 500
                
        except subprocess.TimeoutExpired:
            return jsonify({
                'success': False,
                'error': 'レポート生成がタイムアウトしました（60秒）'
            }), 500
        except Exception as e:
            print(f"❌ 詳細レポート生成でエラー: {e}")
            return jsonify({
                'success': False,
                'error': f'予期しないエラー: {str(e)}'
            }), 500
    
    @app.route('/api/reports/<path:filename>')
    def serve_report(filename):
        """生成されたレポートファイルを配信"""
        try:
            from flask import send_from_directory
            from investment_toolkit.utilities.paths import get_or_create_reports_config

            _reports_config = get_or_create_reports_config()
            reports_dir = _reports_config.base_dir
            
            # セキュリティチェック: パストラバーサル攻撃を防ぐ
            safe_path = reports_dir / filename
            if not str(safe_path).startswith(str(reports_dir.absolute())):
                return jsonify({'error': '不正なファイルパスです'}), 400
            
            if not safe_path.exists():
                return jsonify({'error': 'ファイルが見つかりません'}), 404
            
            # ファイルの親ディレクトリと相対パスを取得
            relative_path = safe_path.relative_to(reports_dir)
            parent_dir = safe_path.parent
            filename_only = safe_path.name
            
            print(f"📄 レポート配信: {relative_path}")
            
            return send_from_directory(
                str(parent_dir), 
                filename_only,
                as_attachment=False,
                mimetype='text/html'
            )
            
        except Exception as e:
            print(f"❌ レポート配信エラー: {e}")
            return jsonify({'error': f'レポート配信エラー: {str(e)}'}), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """ヘルスチェック"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'watchlist-api'
        })
    
    @app.route('/')
    def index():
        """API概要ページ"""
        api_docs = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Watchlist API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }
                .method { display: inline-block; padding: 2px 8px; border-radius: 3px; color: white; font-weight: bold; }
                .post { background: #28a745; }
                .get { background: #007bff; }
                .delete { background: #dc3545; }
                code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>📊 Watchlist API</h1>
            <p>ウォッチリスト機能のAPIエンドポイント一覧</p>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /api/watchlist/add</h3>
                <p>ウォッチリストに銘柄を追加</p>
                <p><strong>リクエスト例:</strong></p>
                <pre><code>[{"symbol": "AAPL", "analysisType": "top_stocks", "metadata": {...}}]</code></pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /api/watchlist</h3>
                <p>現在のウォッチリストを取得</p>
                <p><strong>パラメータ:</strong> <code>analysis_type</code> (オプション)</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /api/watchlist/performance</h3>
                <p>ウォッチリストのパフォーマンスサマリーを取得</p>
                <p><strong>パラメータ:</strong> <code>analysis_type, days_back</code> (オプション)</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /api/watchlist/stock/{symbol}</h3>
                <p>特定銘柄のパフォーマンス履歴を取得</p>
                <p><strong>パラメータ:</strong> <code>analysis_type</code> (オプション)</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method delete">DELETE</span> /api/watchlist/remove/{stock_id}</h3>
                <p>ウォッチリストから銘柄を削除</p>
                <p><strong>リクエスト例:</strong></p>
                <pre><code>{"reason": "削除理由"}</code></pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /api/generate_detailed_report</h3>
                <p>指定銘柄の詳細分析レポートを生成</p>
                <p><strong>リクエスト例:</strong></p>
                <pre><code>{"symbol": "NVDA"}</code></pre>
                <p><strong>説明:</strong> generate_single_stock_report.py を実行して詳細なレポートHTMLを生成し、ブラウザで表示可能なファイルパスを返します。</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /api/reports/{filename}</h3>
                <p>生成されたレポートファイルを配信</p>
                <p><strong>リクエスト例:</strong></p>
                <pre><code>/api/reports/individual_stocks/AAPL_Apple_Inc./AAPL_comprehensive_report_20250731_065402.html</code></pre>
                <p><strong>説明:</strong> reports ディレクトリ内のHTMLレポートをHTTP経由で配信します。</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /api/health</h3>
                <p>ヘルスチェック</p>
            </div>
            
            <hr>
            <p><small>生成時刻: {{ timestamp }}</small></p>
        </body>
        </html>
        """
        return render_template_string(api_docs, timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# シミュレーションモード（Flask未インストール時）
class SimulationAPI:
    """Flask未インストール時のシミュレーションモード"""
    
    def __init__(self):
        self.local_storage = []
    
    def add_to_watchlist(self, stocks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ローカルストレージに銘柄を追加"""
        timestamp = datetime.now().isoformat()
        
        for stock in stocks_data:
            stock_entry = {
                'id': len(self.local_storage) + 1,
                'symbol': stock['symbol'],
                'analysis_type': stock['analysisType'],
                'metadata': stock['metadata'],
                'added_timestamp': timestamp,
                'status': 'active'
            }
            self.local_storage.append(stock_entry)
        
        return {
            'success_count': len(stocks_data),
            'failure_count': 0,
            'total_count': len(stocks_data),
            'errors': []
        }
    
    def get_watchlist(self) -> List[Dict[str, Any]]:
        """ローカルストレージからウォッチリストを取得"""
        return [stock for stock in self.local_storage if stock['status'] == 'active']
    
    def remove_from_watchlist(self, stock_id: int) -> bool:
        """ローカルストレージから銘柄を削除"""
        for stock in self.local_storage:
            if stock['id'] == stock_id:
                stock['status'] = 'removed'
                return True
        return False


def run_server(host='127.0.0.1', port=5000, debug=False):
    """APIサーバーを起動"""
    if FLASK_AVAILABLE:
        print(f"🚀 ウォッチリストAPIサーバーを起動中...")
        print(f"   URL: http://{host}:{port}")
        print(f"   API ドキュメント: http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)
    else:
        print("⚠️ Flask が利用できないため、シミュレーションモードで動作中")
        print("   pip install flask flask-cors でFlaskをインストールしてください")
        sim_api = SimulationAPI()
        
        # シミュレーション例
        sample_stocks = [
            {
                'symbol': 'AAPL',
                'analysisType': 'top_stocks',
                'metadata': {'score': 85.5, 'price': 175.0, 'rsi': 45.2}
            }
        ]
        
        result = sim_api.add_to_watchlist(sample_stocks)
        print(f"   シミュレーション追加結果: {result}")
        
        watchlist = sim_api.get_watchlist()
        print(f"   シミュレーションウォッチリスト: {len(watchlist)}件")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ウォッチリスト API サーバー')
    parser.add_argument('--host', default='127.0.0.1', help='ホスト (デフォルト: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='ポート (デフォルト: 5000)')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, debug=args.debug) 