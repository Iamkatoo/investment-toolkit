import logging
from pushover_complete import PushoverAPI
from investment_toolkit.utilities.config import PUSHOVER_TOKEN, PUSHOVER_USER_KEY

logger = logging.getLogger(__name__)

class NotificationManager:
    """
    Pushoverを使用した通知マネージャー
    """
    
    def __init__(self):
        """
        NotificationManagerのコンストラクタ
        """
        self.pushover = PushoverAPI(PUSHOVER_TOKEN)
        self.user_key = PUSHOVER_USER_KEY
        
    def send_notification(self, title, message, priority=0):
        """
        通知を送信する
        
        パラメータ:
            title (str): 通知のタイトル
            message (str): 通知の本文
            priority (int): 通知の優先度 (-2: 最低, -1: 低, 0: 通常, 1: 高, 2: 緊急)
        
        戻り値:
            bool: 送信成功時はTrue、失敗時はFalse
        """
        try:
            # 通知送信
            self.pushover.send_message(
                user=self.user_key,
                message=message,
                title=title,
                priority=priority
            )
            logger.info(f"通知を送信しました: {title}")
            return True
        except Exception as e:
            logger.error(f"通知の送信に失敗しました: {e}")
            return False
            
    def send_start_notification(self, script_name):
        """
        スクリプト開始通知を送信する
        
        パラメータ:
            script_name (str): 実行中のスクリプト名
        """
        title = "🚀 バッチ処理開始"
        message = f"{script_name}の実行を開始しました。"
        return self.send_notification(title, message)
        
    def send_completion_notification(self, script_name, summary, api_usage=None):
        """
        スクリプト完了通知を送信する
        
        パラメータ:
            script_name (str): 実行中のスクリプト名
            summary (str): 実行結果のサマリー
            api_usage (dict, optional): API使用量の情報
        """
        title = "✅ 投資分析スクリプト実行完了"
        message = f"{script_name}の処理が正常に終了しました。\n\n{summary}"
        
        # API使用量の情報があれば追加
        if api_usage:
            message += f"\n\n【API通信量】\n総データ量: {api_usage['total_data_size_formatted']} ({api_usage['request_count']}リクエスト)"
            
        return self.send_notification(title, message)
        
    def send_indicators_completion_notification(self, summary, api_usage=None):
        """
        経済指標・センチメント・為替・FREDデータ更新完了通知を送信する
        
        パラメータ:
            summary (str): 実行結果のサマリー
            api_usage (dict, optional): API使用量の情報
        """
        title = "✅ バッチ処理完了"
        message = f"経済指標、センチメント、為替レート、FREDデータの更新が完了しました。\n\n{summary}"
        
        # API使用量の情報があれば追加
        if api_usage:
            message += f"\n\n【API通信量】\n総データ量: {api_usage['total_data_size_formatted']} ({api_usage['request_count']}リクエスト)"
            
        return self.send_notification(title, message)
        
    def send_anomaly_notification(self, anomaly_code, severity, observed_value, baseline, message, run_id=None):
        """
        異常検知通知を送信する
        
        パラメータ:
            anomaly_code (str): 異常コード
            severity (str): 重要度（WARN, ALERT, CRITICAL）
            observed_value: 観測値
            baseline: ベースライン値
            message (str): 詳細メッセージ
            run_id (str, optional): 実行ID
        """
        # 重要度に応じて優先度を設定
        priority_map = {
            "WARN": 0,
            "ALERT": 1,
            "CRITICAL": 2
        }
        priority = priority_map.get(severity, 0)
        
        # 絵文字を重要度に応じて設定
        emoji_map = {
            "WARN": "⚠️",
            "ALERT": "🚨",
            "CRITICAL": "🔥"
        }
        emoji = emoji_map.get(severity, "📊")
        
        title = f"{emoji} [MonthlyUpdate] {severity}: {anomaly_code}"
        
        notification_message = f"""異常が検知されました:

【詳細】
異常コード: {anomaly_code}
重要度: {severity}
観測値: {observed_value}
ベースライン: {baseline}

【説明】
{message}"""
        
        if run_id:
            notification_message += f"\n\n実行ID: {run_id}"
        
        return self.send_notification(title, notification_message, priority)
    
    def send_monthly_update_start_notification(self, script_name):
        """
        月次更新開始通知を送信する（高優先度）
        
        パラメータ:
            script_name (str): 実行中のスクリプト名
        """
        title = "🗓️ 月次更新開始"
        message = f"{script_name}の月次更新を開始しました。"
        return self.send_notification(title, message, priority=1)
        
    def send_monthly_update_completion_notification(self, script_name, summary, api_usage=None):
        """
        月次更新完了通知を送信する（高優先度）
        
        パラメータ:
            script_name (str): 実行中のスクリプト名
            summary (str): 実行結果のサマリー
            api_usage (dict, optional): API使用量の情報
        """
        title = "✅ 月次更新完了"
        message = f"{script_name}の月次更新が完了しました。\n\n{summary}"
        
        # API使用量の情報があれば追加
        if api_usage:
            message += f"\n\n【API通信量】\n総データ量: {api_usage['total_data_size_formatted']} ({api_usage['request_count']}リクエスト)"
            
        return self.send_notification(title, message, priority=1)

    def send_tail(self, logger_name, lines=30):
        """
        ログファイルの最新部分を通知として送信する
        
        パラメータ:
            logger_name (str): ロガー名
            lines (int): 取得する行数
        """
        from pathlib import Path
        import os

        log_dir = Path(os.getenv("LOG_DIR", "./logs")).expanduser().resolve()
        f = log_dir / f"{logger_name}.log"
        if f.exists():
            tail = "\n".join(f.read_text().splitlines()[-lines:])
            self.send_notification(f"📄 {logger_name} tail", tail) 