# J-Quants API パッケージ

日本株データ取得のためのJ-Quants API統合モジュールです。FMPから取得していた日本株データをJ-Quantsに移行するために作成されました。

## 機能

- **認証管理**: REFRESH_TOKENとidTokenの自動取得・管理
- **日足データ取得**: 指定日または期間の日足データ取得
- **データベース統合**: 既存のデータベーススキーマとの互換性を保った格納
- **エラーハンドリング**: リトライ機能付きの堅牢なAPI呼び出し

## セットアップ

### 1. 環境変数設定

`.env`ファイルに以下の設定を追加してください：

```env
JQUANTS_EMAIL=your_email@example.com
JQUANTS_PASSWORD=your_password
```

### 2. データベースセットアップ

スキーマとテーブルを作成します：

```bash
psql -h localhost -U your_user -d investment -f db/create_jquants_schema.sql
```

## 使用方法

### コマンドライン実行

```bash
# 今日のデータを取得
python -m src.j_quants_api.daily_price_fetcher

# 特定の日付のデータを取得
python -m src.j_quants_api.daily_price_fetcher --date 2025-01-15

# 期間指定でデータを取得
python -m src.j_quants_api.daily_price_fetcher --start-date 2025-01-01 --end-date 2025-01-15

# ログレベルを指定
python -m src.j_quants_api.daily_price_fetcher --date 2025-01-15 --log-level DEBUG
```

### プログラムから使用

```python
from datetime import date
from investment_analysis.j_quants_api import JQuantsDailyPriceFetcher

# フェッチャーを初期化
fetcher = JQuantsDailyPriceFetcher()

# 今日のデータを取得・保存
result = fetcher.fetch_and_save_daily_prices(date.today())
print(f"取得: {result['fetched_count']} 件, 保存: {result['saved_count']} 件")

# 期間指定で取得
from datetime import date, timedelta
start_date = date.today() - timedelta(days=7)
end_date = date.today()
results = fetcher.fetch_date_range(start_date, end_date)
```

### 認証のみ使用

```python
from investment_analysis.j_quants_api import get_auth

# 認証インスタンス取得
auth = get_auth()

# IDトークン取得
id_token = auth.get_id_token()

# 認証ヘッダー取得
headers = auth.get_auth_headers()
```

## データ構造

J-Quantsから取得したデータは`j_quants_data.daily_prices`テーブルに格納されます。

主要フィールド：
- `symbol`: 銘柄コード（例: "1332.T"）
- `date`: 取引日
- `open/high/low/close`: 調整後株価
- `volume`: 出来高
- `turnover_value`: 売買代金
- `adjustment_factor`: 調整係数

## 統合

### jp_close.shでの実行

日本株の日次処理バッチ（`batch/jp_close.sh`）に組み込まれています：

```bash
cd /path/to/Investment
./batch/jp_close.sh
```

FMPデータとの並行実行期間中は、両方のデータソースから取得されます。

## トラブルシューティング

### 認証エラー

- 環境変数`JQUANTS_EMAIL`と`JQUANTS_PASSWORD`が正しく設定されているか確認
- J-Quantsのアカウントが有効か確認

### API制限エラー

- レート制限により自動的に待機時間が挿入されます
- エラーが継続する場合は時間をおいて再実行

### データベースエラー

- PostgreSQLサーバーが起動しているか確認
- データベース接続設定（`DB_HOST`, `DB_USER`等）を確認
- スキーマが正しく作成されているか確認

## ログ

ログレベルは以下から選択できます：
- `DEBUG`: 詳細なデバッグ情報
- `INFO`: 一般的な実行情報（推奨）
- `WARNING`: 警告とエラー
- `ERROR`: エラーのみ

## 制限事項

- 土日祝日のデータは取得されません（取引所休業日）
- 過去データの一括取得時はAPI制限を考慮して時間がかかります
- J-Quants API仕様変更時は対応が必要な場合があります 