# 銘柄の履歴データを取得するメソッドを修正し、詳細なログ出力を追加します
def get_historical_price_data(self, symbol, from_date, to_date=None):
    """
    指定した銘柄の株価履歴データを取得する
    
    Parameters
    ----------
    symbol : str
        銘柄コード
    from_date : str
        開始日 (YYYY-MM-DD)
    to_date : str, optional
        終了日 (YYYY-MM-DD), デフォルトはNone (現在日付が使用される)
    
    Returns
    -------
    pd.DataFrame or None
        株価履歴データ、取得できない場合はNone
    """
    self.logger.debug(f"{symbol}の株価履歴データ取得を開始します (期間: {from_date} ～ {to_date or '現在'})")
    
    # 問題のある銘柄かどうかのチェック（FMP APIクラスで管理されている問題銘柄リストを確認）
    if hasattr(self.fmp_api, 'problematic_symbols') and symbol in self.fmp_api.problematic_symbols:
        self.logger.warning(f"{symbol}は問題銘柄リストに登録されているためスキップします")
        return None
    
    try:
        # 日本株の場合は旧エンドポイントを先に試す
        if is_japanese_stock(symbol):
            self.logger.debug(f"{symbol}は日本株と判断されました。旧エンドポイントを試行します")
            historical_data = self._get_historical_price_data_legacy(symbol, from_date, to_date)
            if historical_data is not None and not historical_data.empty:
                self.logger.debug(f"{symbol}の株価履歴データを旧エンドポイントから取得しました（{len(historical_data)}件）")
                return historical_data
            else:
                self.logger.warning(f"{symbol}の株価履歴データ取得が旧エンドポイントで失敗しました。標準エンドポイントを試行します")
        
        # 標準エンドポイントを使用
        params = {
            'symbol': symbol,
            'from': from_date
        }
        if to_date:
            params['to'] = to_date
            
        self.logger.debug(f"{symbol}の株価履歴データを標準エンドポイントから取得を試みます: パラメータ={params}")
        
        historical_json = self.fmp_api.fetch_data('/historical-price-full/' + symbol, params)
        
        if historical_json is None:
            self.logger.error(f"{symbol}の株価履歴データが取得できませんでした（標準エンドポイント）")
            
            # サブスクリプションエラーなどで失敗した場合、日本株でなくても旧エンドポイントを試してみる
            self.logger.warning(f"{symbol}の株価履歴データ取得でエラーが発生しました。旧エンドポイントを試行します。")
            historical_data = self._get_historical_price_data_legacy(symbol, from_date, to_date)
            if historical_data is not None and not historical_data.empty:
                self.logger.debug(f"{symbol}の株価履歴データを旧エンドポイントから取得しました（{len(historical_data)}件）")
                return historical_data
                
            self.logger.error(f"{symbol}の株価履歴データが取得できませんでした")
            return None
            
        # レスポンスのデータ構造を詳細にログ出力
        if isinstance(historical_json, dict):
            self.logger.debug(f"{symbol}のレスポンス構造: {list(historical_json.keys())}")
            
            if 'historical' not in historical_json or not historical_json['historical']:
                self.logger.warning(f"{symbol}の履歴データが存在しないか空です")
                return None
                
            historical_data = pd.DataFrame(historical_json['historical'])
            self.logger.debug(f"{symbol}の株価履歴データを取得しました（{len(historical_data)}件）")
            return historical_data
        else:
            self.logger.error(f"{symbol}のレスポンスが予期しない形式です: {type(historical_json)}")
            return None
            
    except Exception as e:
        self.logger.error(f"{symbol}の株価履歴データ取得中にエラーが発生しました: {str(e)}")
        traceback.print_exc()
        return None

def _get_historical_price_data_legacy(self, symbol, from_date, to_date=None):
    """
    旧エンドポイントを使用して株価履歴データを取得する（日本株など特定の銘柄用）
    """
    try:
        endpoint = f"/api/v3/historical-price-full/{symbol}"
        params = {
            'from': from_date,
        }
        if to_date:
            params['to'] = to_date
            
        self.logger.debug(f"{symbol}の株価履歴データを旧エンドポイントから取得を試みます: エンドポイント={endpoint}, パラメータ={params}")
        
        # Legacy APIを使用するフラグをTrueに設定
        historical_json = self.fmp_api.fetch_data(endpoint, params, use_legacy=True)
        
        if historical_json is None:
            self.logger.error(f"{symbol}の株価履歴データが旧エンドポイントから取得できませんでした")
            return None
            
        # レスポンスの内容をログに記録
        if isinstance(historical_json, dict):
            self.logger.debug(f"{symbol}の旧エンドポイントレスポンス構造: {list(historical_json.keys())}")
            
            if 'historical' not in historical_json or not historical_json['historical']:
                self.logger.warning(f"{symbol}の旧エンドポイントからの履歴データが存在しないか空です")
                return None
                
            historical_data = pd.DataFrame(historical_json['historical'])
            self.logger.debug(f"{symbol}の株価履歴データを旧エンドポイントから取得しました（{len(historical_data)}件）")
            return historical_data
        else:
            self.logger.error(f"{symbol}の旧エンドポイントからのレスポンスが予期しない形式です: {type(historical_json)}")
            return None
    
    except Exception as e:
        self.logger.error(f"{symbol}の株価履歴データ旧エンドポイントからの取得中にエラーが発生しました: {str(e)}")
        traceback.print_exc()
        return None 