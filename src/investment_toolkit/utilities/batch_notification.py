#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
バッチスクリプト用の通知ユーティリティ

使用方法:
python src/utilities/batch_notification.py --action start --script us_close
python src/utilities/batch_notification.py --action complete --script us_close --log-file log/us_20250115.log
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from investment_analysis.utilities.notification import NotificationManager


class BatchNotificationManager:
    """バッチスクリプト専用の通知マネージャー"""
    
    def __init__(self):
        self.notification = NotificationManager()
        self.script_display_names = {
            "us_close": "🇺🇸 US市場クローズ処理",
            "jp_close": "🇯🇵 JP市場クローズ処理", 
            "weekly_update": "📅 週次更新処理",
            "monthly_update": "🗓️ 月次更新処理"
        }
    
    def send_start_notification(self, script_name):
        """
        バッチスクリプト開始通知
        
        Args:
            script_name (str): スクリプト名
        """
        display_name = self.script_display_names.get(script_name, f"📊 {script_name}")
        title = "🚀 バッチ処理開始"
        message = f"{display_name}を開始しました。\n開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.notification.send_notification(title, message, priority=0)
    
    def send_completion_notification(self, script_name, log_file_path=None):
        """
        バッチスクリプト完了通知（ログファイルから結果を解析）
        
        Args:
            script_name (str): スクリプト名
            log_file_path (str): ログファイルのパス
        """
        display_name = self.script_display_names.get(script_name, f"📊 {script_name}")
        
        # ログファイルから実行結果を解析
        summary = self._analyze_log_file(log_file_path) if log_file_path else "ログファイルが指定されていません"
        
        title = "✅ バッチ処理完了"
        message = f"{display_name}が完了しました。\n\n{summary}"
        
        return self.notification.send_notification(title, message, priority=0)
    
    def _analyze_log_file(self, log_file_path):
        """
        ログファイルを解析して実行サマリーを作成
        
        Args:
            log_file_path (str): ログファイルのパス
            
        Returns:
            str: 実行サマリー
        """
        if not os.path.exists(log_file_path):
            return f"❌ ログファイルが見つかりません: {log_file_path}"
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # ログから重要な情報を抽出
            start_time = self._extract_start_time(log_content)
            end_time = self._extract_end_time(log_content)
            errors = self._extract_errors(log_content)
            success_info = self._extract_success_info(log_content)
            
            # 実行時間を計算
            duration = self._calculate_duration(start_time, end_time)
            
            summary_parts = []
            
            # 実行時間
            if duration:
                summary_parts.append(f"⏱️ 実行時間: {duration}")
            
            # 成功情報
            if success_info:
                summary_parts.append(f"📊 処理結果:\n{success_info}")
            
            # エラー情報
            if errors:
                summary_parts.append(f"⚠️ エラー・警告:\n{errors}")
            else:
                summary_parts.append("✅ エラーなし")
            
            return "\n\n".join(summary_parts)
            
        except Exception as e:
            return f"❌ ログファイル解析エラー: {str(e)}"
    
    def _extract_start_time(self, log_content):
        """ログから開始時刻を抽出"""
        pattern = r"=== .+ batch start (.+) ==="
        match = re.search(pattern, log_content)
        return match.group(1) if match else None
    
    def _extract_end_time(self, log_content):
        """ログから終了時刻を抽出"""
        pattern = r"=== .+ batch end (.+) ==="
        match = re.search(pattern, log_content)
        return match.group(1) if match else None
    
    def _extract_errors(self, log_content):
        """ログからエラー情報を抽出"""
        error_patterns = [
            r"Error in ([^,]+)",
            r"ERROR\s*:\s*(.+)",
            r"CRITICAL\s*:\s*(.+)",
            r"WARNING\s*:\s*(.+)"
        ]
        
        errors = []
        for pattern in error_patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            for match in matches[:5]:  # 最大5個まで
                errors.append(f"• {match.strip()}")
        
        return "\n".join(errors) if errors else None
    
    def _extract_success_info(self, log_content):
        """ログから成功情報を抽出"""
        success_patterns = [
            (r"処理シンボル数\s*:\s*(\d+)", "処理銘柄数"),
            (r"成功率\s*:\s*([\d.]+)%", "成功率"),
            (r"リクエスト総数\s*:\s*(\d+)", "API リクエスト数"),
            (r"通信量 \(MB\)\s*:\s*([\d.]+)", "通信量 (MB)"),
            (r"(\d+)件の.+データを取得", "データ取得件数"),
            (r"(\d+)銘柄.+更新", "更新銘柄数")
        ]
        
        success_info = []
        for pattern, label in success_patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            if matches:
                # 最後のマッチを使用（最新の値）
                value = matches[-1]
                success_info.append(f"• {label}: {value}")
        
        return "\n".join(success_info) if success_info else "処理内容の詳細は不明"
    
    def _calculate_duration(self, start_time, end_time):
        """開始時刻と終了時刻から実行時間を計算"""
        if not start_time or not end_time:
            return None
        
        try:
            # 時刻フォーマットの例: "Mon Jan 15 12:34:56 JST 2025"
            start_dt = datetime.strptime(start_time.strip(), "%a %b %d %H:%M:%S %Z %Y")
            end_dt = datetime.strptime(end_time.strip(), "%a %b %d %H:%M:%S %Z %Y")
            
            duration = end_dt - start_dt
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            seconds = duration.seconds % 60
            
            if hours > 0:
                return f"{hours}時間{minutes}分{seconds}秒"
            elif minutes > 0:
                return f"{minutes}分{seconds}秒"
            else:
                return f"{seconds}秒"
                
        except (ValueError, AttributeError):
            return None


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="バッチスクリプト通知ユーティリティ")
    parser.add_argument("--action", required=True, choices=["start", "complete", "custom"],
                       help="実行アクション")
    parser.add_argument("--script", required=True, 
                       help="スクリプト名 (us_close, jp_close, weekly_update, monthly_update)")
    parser.add_argument("--log-file",
                       help="ログファイルのパス（完了通知時に使用）")
    parser.add_argument("--message",
                       help="カスタム通知のメッセージ（customアクション時に使用）")
    parser.add_argument("--title",
                       help="カスタム通知のタイトル（customアクション時に使用）")
    
    args = parser.parse_args()
    
    notification_manager = BatchNotificationManager()
    
    if args.action == "start":
        success = notification_manager.send_start_notification(args.script)
        if success:
            print(f"✅ 開始通知を送信しました: {args.script}")
        else:
            print(f"❌ 開始通知の送信に失敗しました: {args.script}")
            sys.exit(1)
    
    elif args.action == "complete":
        success = notification_manager.send_completion_notification(args.script, args.log_file)
        if success:
            print(f"✅ 完了通知を送信しました: {args.script}")
        else:
            print(f"❌ 完了通知の送信に失敗しました: {args.script}")
            sys.exit(1)

    elif args.action == "custom":
        if not args.message:
            print("❌ customアクションにはmessageパラメータが必要です")
            sys.exit(1)

        title = args.title or "カスタム通知"
        success = notification_manager.notification.send_notification(title, args.message, priority=0)
        if success:
            print(f"✅ カスタム通知を送信しました: {title}")
        else:
            print(f"❌ カスタム通知の送信に失敗しました: {title}")
            sys.exit(1)


if __name__ == "__main__":
    main()
