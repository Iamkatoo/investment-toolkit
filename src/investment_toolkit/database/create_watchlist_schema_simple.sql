-- ウォッチリスト機能用の軽量スキーマ（大きなJOINを避ける）
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

-- パフォーマンストラッキングテーブル
CREATE TABLE IF NOT EXISTS watchlist.performance_tracking (
    id SERIAL PRIMARY KEY,
    tracked_stock_id INTEGER REFERENCES watchlist.tracked_stocks(id) ON DELETE CASCADE,
    analysis_date DATE NOT NULL,
    price DECIMAL(10,2),
    rsi_14 DECIMAL(5,2),
    total_score DECIMAL(8,2),
    value_score DECIMAL(8,2),
    growth_score DECIMAL(8,2),
    quality_score DECIMAL(8,2),
    momentum_score DECIMAL(8,2),
    macro_sector_score DECIMAL(8,2),
    price_change_pct DECIMAL(8,4),
    rsi_change DECIMAL(8,4),
    score_change DECIMAL(8,4),
    days_since_added INTEGER,
    market_cap BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tracked_stock_id, analysis_date)
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

-- 軽量ビュー：基本的なウォッチリスト情報のみ（重いJOINを避ける）
CREATE OR REPLACE VIEW watchlist.vw_simple_watchlist AS
SELECT 
    w.id,
    w.symbol,
    w.added_date,
    w.analysis_type,
    w.analysis_category,
    w.analysis_metadata,
    w.notes,
    (CURRENT_DATE - w.added_date) as days_since_added,
    -- 追加時の価格とRSI（メタデータから取得）
    (w.analysis_metadata->>'price')::numeric as initial_price,
    (w.analysis_metadata->>'rsi')::numeric as initial_rsi,
    (w.analysis_metadata->>'score')::numeric as initial_score
FROM watchlist.tracked_stocks w
WHERE w.is_active = true
ORDER BY w.added_date DESC;

-- パフォーマンストラッキング更新関数（型キャスト修正版）
CREATE OR REPLACE FUNCTION watchlist.update_performance_tracking()
RETURNS INTEGER AS $$
DECLARE
    rows_inserted INTEGER := 0;
BEGIN
    INSERT INTO watchlist.performance_tracking 
    (tracked_stock_id, analysis_date, price, rsi_14, total_score, 
     value_score, growth_score, quality_score, momentum_score, macro_sector_score,
     price_change_pct, rsi_change, score_change, days_since_added, market_cap)
    SELECT 
        w.id as tracked_stock_id,
        CURRENT_DATE as analysis_date,
        vm.close as price,
        vm.rsi_14,
        ds.total_score,
        ds.value_score,
        ds.growth_score,
        ds.quality_score,
        ds.momentum_score,
        ds.macro_sector_score,
        ROUND((((vm.close::numeric - (w.analysis_metadata->>'price')::numeric) / (w.analysis_metadata->>'price')::numeric * 100)), 2) as price_change_pct,
        ROUND(((vm.rsi_14::numeric - (w.analysis_metadata->>'rsi')::numeric)), 2) as rsi_change,
        ROUND(((ds.total_score::numeric - (w.analysis_metadata->>'score')::numeric)), 2) as score_change,
        (CURRENT_DATE - w.added_date) as days_since_added,
        vm.market_cap
    FROM watchlist.tracked_stocks w
    LEFT JOIN backtest_results.vw_daily_master vm 
        ON w.symbol = vm.symbol 
        AND vm.date = (SELECT MAX(date) FROM backtest_results.vw_daily_master)
    LEFT JOIN backtest_results.daily_scores ds
        ON w.symbol = ds.symbol 
        AND ds.date = (SELECT MAX(date) FROM backtest_results.daily_scores)
    WHERE w.is_active = true
    AND NOT EXISTS (
        SELECT 1 FROM watchlist.performance_tracking pt 
        WHERE pt.tracked_stock_id = w.id AND pt.analysis_date = CURRENT_DATE
    );
    
    GET DIAGNOSTICS rows_inserted = ROW_COUNT;
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;

-- 自動更新のためのトリガー関数
CREATE OR REPLACE FUNCTION watchlist.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- トリガー作成
DROP TRIGGER IF EXISTS update_tracked_stocks_updated_at ON watchlist.tracked_stocks;
CREATE TRIGGER update_tracked_stocks_updated_at 
    BEFORE UPDATE ON watchlist.tracked_stocks 
    FOR EACH ROW EXECUTE FUNCTION watchlist.update_updated_at_column(); 