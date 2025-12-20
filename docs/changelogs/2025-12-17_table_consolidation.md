# テーブル統合: active_trading_symbols → symbol_status

## 変更日
2025-12-17

## 概要
`fmp_data.active_trading_symbols`テーブルの機能を`fmp_data.symbol_status`テーブルに統合しました。これにより、データの重複が解消され、銘柄管理が一元化されました。

## 変更内容

### 修正ファイル
- `investment-toolkit/src/investment_toolkit/data/stable_api_manager.py`

### 主な変更点

#### 1. `fetch_and_store_actively_trading_list()` メソッド
**変更前**:
- actively-trading-list APIから取得したデータを`active_trading_symbols`テーブルに保存

**変更後**:
- actively-trading-list APIから取得したデータを`symbol_status`テーブルに直接保存
- 基本情報（symbol, name）のみを保存し、exchange等の詳細は後続ステップで更新

#### 2. `_get_actively_trading_baseline()` メソッド
**変更前**:
- `active_trading_symbols.updated_at`から過去3ヶ月の平均件数を計算

**変更後**:
- `symbol_status.last_updated`から過去3ヶ月の平均件数を計算

#### 3. `fetch_and_store_company_profiles()` メソッド
**変更前**:
- `active_trading_symbols`テーブルから銘柄リストを取得

**変更後**:
- `symbol_status`テーブルから銘柄リストを取得

## データフロー（変更後）

```
月次更新プロセス:
1. actively-trading-list API取得
   ↓
2. symbol_statusに基本情報を保存
   ↓
3. 各銘柄のcompany profileを取得
   ↓
4. symbol_statusをexchange情報で更新（is_active判定）
   ↓
5. 取引所制限によるmanually_deactivated更新
   ↓
6. is_active=trueの銘柄の従業員数データ更新
```

## アクティブ銘柄の取得方法

### 変更前
```sql
SELECT symbol FROM fmp_data.active_trading_symbols
ORDER BY symbol;
```

### 変更後
```sql
SELECT symbol FROM fmp_data.symbol_status
WHERE is_active = true AND manually_deactivated = false
ORDER BY symbol;
```

## 影響範囲

### 修正済み
- ✅ `investment-toolkit/src/investment_toolkit/data/stable_api_manager.py`
- ✅ ドキュメント更新（database_structure.md, component_map.md）

### 影響なし
- ✅ `investment-workspace` - active_trading_symbolsの使用箇所なし

## 後方互換性

既存のデータフローは維持されています。`symbol_status`テーブルは既に存在しており、今回の変更で以下が実現されました:

1. データの一元管理
2. 重複の排除
3. クエリの単純化

## テスト推奨事項

1. **月次更新のテスト実行**
   ```bash
   cd investment-toolkit
   python -m investment_toolkit.data.stable_api_manager --test --limit 10
   ```

2. **データ整合性確認**
   ```sql
   -- symbol_statusのレコード数確認
   SELECT COUNT(*) FROM fmp_data.symbol_status;

   -- アクティブ銘柄数確認
   SELECT COUNT(*) FROM fmp_data.symbol_status
   WHERE is_active = true AND manually_deactivated = false;
   ```

3. **異常検知の動作確認**
   - ベースライン計算が正しく動作するか確認
   - 異常検知通知が正常に送信されるか確認

## テーブル削除手順

### active_trading_symbolsテーブルの削除

統合が完了したため、以下のコマンドでテーブルを削除してください:

```sql
-- オプション: バックアップ作成（念のため）
CREATE TABLE fmp_data.active_trading_symbols_backup_20251217 AS
SELECT * FROM fmp_data.active_trading_symbols;

-- テーブル削除
DROP TABLE fmp_data.active_trading_symbols CASCADE;

-- 確認
\dt fmp_data.active_trading_symbols
-- "Did not find any relation named "active_trading_symbols"" と表示されればOK
```

### 削除後の確認

1. **データベース構造ドキュメントの更新**
   ```bash
   python scripts/generate_db_documentation.py
   ```

2. **symbol_statusが正常に動作していることを確認**
   ```sql
   SELECT COUNT(*) FROM fmp_data.symbol_status;
   SELECT COUNT(*) FROM fmp_data.symbol_status
   WHERE is_active = true AND manually_deactivated = false;
   ```

## 関連Issue/PR
- テーブル統合によるデータ管理の簡素化
- symbol_statusテーブルへの機能集約

## 備考
- この変更により、メンテナンスコストが削減されました
- データの一貫性が向上しました
- 将来的な拡張が容易になりました
