-- ウォッチリスト機能用のスキーマとテーブル作成
-- 既存のデータベースには一切影響しない新しいスキーマ

-- ウォッチリストスキーマ作成
CREATE SCHEMA IF NOT EXISTS watchlist;

-- ウォッチリスト銘柄テーブル
CREATE TABLE IF NOT EXISTS watchlist.tracked_stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    added_date DATE NOT NULL DEFAULT CURRENT_DATE,
    analysis_type VARCHAR(50) NOT NULL, -- 'top_stocks', 'rsi35_below', 'momentum_breakout' など
    analysis_category VARCHAR(50), -- 'high_score', 'oversold_growth', 'momentum_play' など
    added_reason TEXT,
    -- 追加時の市場データ（JSON形式で保存）
    analysis_metadata JSONB DEFAULT '{}', -- スコア、RSI、価格、その他の分析固有データ
    market_conditions JSONB DEFAULT '{}', -- 追加時の市場状況
    is_active BOOLEAN DEFAULT true,
    removed_date DATE,
    removal_reason TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- パフォーマンス追跡テーブル
CREATE TABLE IF NOT EXISTS watchlist.performance_tracking (
    id SERIAL PRIMARY KEY,
    tracked_stock_id INTEGER REFERENCES watchlist.tracked_stocks(id) ON DELETE CASCADE,
    analysis_date DATE NOT NULL,
    price NUMERIC(20,6),
    rsi_14 NUMERIC(10,4),
    total_score NUMERIC(10,4),
    value_score NUMERIC(10,4),
    growth_score NUMERIC(10,4),
    quality_score NUMERIC(10,4),
    momentum_score NUMERIC(10,4),
    macro_sector_score NUMERIC(10,4),
    -- 追加日からの変化率
    price_change_pct NUMERIC(10,4),
    rsi_change NUMERIC(10,4),
    score_change NUMERIC(10,4),
    days_since_added INTEGER,
    market_cap NUMERIC(20,2),
    volume NUMERIC(20,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 分析タイプ別統計テーブル
CREATE TABLE IF NOT EXISTS watchlist.analysis_performance (
    id SERIAL PRIMARY KEY,
    analysis_type VARCHAR(50) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_picks INTEGER DEFAULT 0,
    winners INTEGER DEFAULT 0,
    losers INTEGER DEFAULT 0,
    avg_return_pct NUMERIC(10,4),
    win_rate NUMERIC(10,4),
    best_performer VARCHAR(20),
    worst_performer VARCHAR(20),
    max_gain_pct NUMERIC(10,4),
    max_loss_pct NUMERIC(10,4),
    analysis_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- インデックス作成（パフォーマンス向上）
CREATE INDEX IF NOT EXISTS idx_tracked_stocks_symbol ON watchlist.tracked_stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_tracked_stocks_analysis_type ON watchlist.tracked_stocks(analysis_type);
CREATE INDEX IF NOT EXISTS idx_tracked_stocks_added_date ON watchlist.tracked_stocks(added_date);
CREATE INDEX IF NOT EXISTS idx_tracked_stocks_active ON watchlist.tracked_stocks(is_active);

CREATE INDEX IF NOT EXISTS idx_performance_tracking_stock_id ON watchlist.performance_tracking(tracked_stock_id);
CREATE INDEX IF NOT EXISTS idx_performance_tracking_date ON watchlist.performance_tracking(analysis_date);

CREATE INDEX IF NOT EXISTS idx_analysis_performance_type ON watchlist.analysis_performance(analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_performance_period ON watchlist.analysis_performance(period_start, period_end);

-- ビュー：アクティブなウォッチリスト銘柄の現在状況
CREATE OR REPLACE VIEW watchlist.vw_current_watchlist AS
SELECT 
    w.id,
    w.symbol,
    w.added_date,
    w.analysis_type,
    w.analysis_category,
    w.analysis_metadata,
    w.notes,
    vm.close as current_price,
    vm.rsi_14 as current_rsi,
    ds.total_score as current_score,
    -- 追加日からの変化計算
    ROUND(((vm.close::numeric - (w.analysis_metadata->>'price')::numeric) / (w.analysis_metadata->>'price')::numeric * 100)::numeric, 2) as price_change_pct,
    ROUND((vm.rsi_14::numeric - (w.analysis_metadata->>'rsi')::numeric)::numeric, 2) as rsi_change,
    ROUND((ds.total_score::numeric - (w.analysis_metadata->>'score')::numeric)::numeric, 2) as score_change,
    (CURRENT_DATE - w.added_date) as days_since_added,
    cp.company_name,
    cg.raw_industry as industry,
    cg.raw_sector as sector
FROM watchlist.tracked_stocks w
LEFT JOIN backtest_results.vw_daily_master vm 
    ON w.symbol = vm.symbol 
    AND vm.date = (SELECT MAX(date) FROM backtest_results.vw_daily_master)
LEFT JOIN backtest_results.daily_scores ds
    ON w.symbol = ds.symbol 
    AND ds.date = (SELECT MAX(date) FROM backtest_results.daily_scores)
LEFT JOIN fmp_data.company_profile cp
    ON w.symbol = cp.symbol
LEFT JOIN reference.company_gics cg
    ON w.symbol = cg.symbol AND cg.is_active = true
WHERE w.is_active = true
ORDER BY w.added_date DESC;

-- 自動更新のためのトリガー関数
CREATE OR REPLACE FUNCTION watchlist.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- トリガー作成
CREATE TRIGGER update_tracked_stocks_updated_at 
    BEFORE UPDATE ON watchlist.tracked_stocks 
    FOR EACH ROW EXECUTE FUNCTION watchlist.update_updated_at_column();

-- サンプルデータ挿入権限の設定（必要に応じて）
-- GRANT USAGE ON SCHEMA watchlist TO your_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA watchlist TO your_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA watchlist TO your_user; 