#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
スコア上位銘柄の詳細分析レポート生成モジュール

機能:
- 日次スコア上位10銘柄の抽出
- 各銘柄の詳細スコア分析
- 財務データの時系列変化分析
- テクニカル分析（価格・SMA・RSI等）
- 週足チャート分析
- 財務・成長指標の深掘り分析
- バリュエーション比較分析
- 投資判断レポートの生成
- シンプルなウォッチリスト機能
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import html
import json
import warnings
warnings.filterwarnings('ignore')

# プロジェクト内のモジュールをインポート
from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

# ウォッチリスト機能は内部で定義済みのため、インポート不要

def add_watchlist_css() -> str:
    """ウォッチリスト機能のCSSを追加"""
    return """
    <style>
        /* ウォッチリスト機能のスタイル */
        .watchlist-controls {
            background-color: #e8f5e8;
            border: 2px solid #27ae60;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }
        .watchlist-checkbox-cell {
            text-align: center;
            vertical-align: middle;
        }
        .watchlist-checkbox {
            transform: scale(1.2);
            margin: 5px;
        }
        .watchlist-status {
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-weight: bold;
        }
        .watchlist-status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .watchlist-status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .watchlist-status.warning {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
    </style>
    """

def generate_watchlist_controls(analysis_type: str) -> str:
    """ウォッチリスト管理コントロールを生成"""
    return f"""
    <div class="watchlist-controls">
        <h3>📋 ウォッチリスト管理</h3>
        <p>チェックボックスで銘柄をウォッチリストに追加/削除できます。</p>
        <div id="watchlist-status" class="watchlist-status" style="display: none;"></div>
        <button onclick="selectAllWatchlist()" style="margin-right: 10px; padding: 8px 16px; background-color: #007bff; color: white; border: none; border-radius: 4px;">全選択</button>
        <button onclick="deselectAllWatchlist()" style="padding: 8px 16px; background-color: #6c757d; color: white; border: none; border-radius: 4px;">全解除</button>
    </div>
    """

def generate_watchlist_checkbox(symbol: str, analysis_type: str, metadata: dict) -> str:
    """ウォッチリスト用チェックボックスを生成"""
    metadata_json = str(metadata).replace("'", '"')
    return f"""
    <input type="checkbox" 
           class="watchlist-checkbox" 
           data-symbol="{symbol}" 
           data-analysis-type="{analysis_type}"
           data-metadata='{metadata_json}'
           onchange="handleWatchlistChange(this)">
    """

def add_watchlist_javascript() -> str:
    """ウォッチリスト機能のJavaScriptを追加"""
    return """
    <script>
        // ウォッチリスト機能のJavaScript（サーバー自動起動機能付き）
        let serverStartAttempted = false;
        
        function showTemporaryMessage(message, type = 'success') {
            const statusDiv = document.getElementById('watchlist-status');
            if (statusDiv) {
                statusDiv.textContent = message;
                statusDiv.className = `watchlist-status ${type}`;
                statusDiv.style.display = 'block';
                
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                }, 3000);
            } else {
                // statusDivがない場合は一時的にアラートで表示
                console.log(`${type}: ${message}`);
            }
        }
        
        async function startApiServerIfNeeded() {
            if (serverStartAttempted) {
                return false; // 既に起動試行済み
            }
            
            try {
                showTemporaryMessage('⚡ APIサーバーを起動中...', 'info');
                
                const response = await fetch('/start_api_server', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                
                if (response.ok) {
                    serverStartAttempted = true;
                    showTemporaryMessage('✅ APIサーバーを起動しました', 'success');
                    // 少し待ってから再試行
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    return true;
                } else {
                    throw new Error('サーバー起動に失敗');
                }
            } catch (error) {
                console.error('API サーバー起動エラー:', error);
                showTemporaryMessage('❌ APIサーバーの自動起動に失敗しました。手動で python start_watchlist_api.py を実行してください。', 'error');
                return false;
            }
        }
        
        function handleWatchlistChange(checkbox) {
            const symbol = checkbox.dataset.symbol;
            const analysisType = checkbox.dataset.analysisType;
            const metadata = JSON.parse(checkbox.dataset.metadata);
            
            if (checkbox.checked) {
                addToWatchlistImmediate(symbol, analysisType, metadata);
            } else {
                removeFromWatchlistImmediate(symbol, analysisType);
            }
        }
        
        async function addToWatchlistImmediate(symbol, analysisType, metadata) {
            const stocksArray = [{
                symbol: symbol,
                analysisType: analysisType,
                metadata: metadata
            }];
            
            try {
                                        const response = await fetch('http://127.0.0.1:5001/api/watchlist/add', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(stocksArray)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showTemporaryMessage(`✅ ${symbol} をウォッチリストに追加しました`, 'success');
                } else {
                    showTemporaryMessage(`❌ ${symbol} の追加に失敗しました`, 'error');
                }
            } catch (error) {
                console.error('API呼び出しエラー:', error);
                
                // API接続エラーの場合、サーバー起動を試行
                showTemporaryMessage('⚠️ APIサーバーに接続できません。自動起動を試行中...', 'warning');
                
                const serverStarted = await startApiServerIfNeeded();
                
                if (serverStarted) {
                    // サーバー起動後に再試行
                    try {
                        const retryResponse = await fetch('http://127.0.0.1:5001/api/watchlist/add', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(stocksArray)
                        });
                        
                        const retryData = await retryResponse.json();
                        
                        if (retryData.success) {
                            showTemporaryMessage(`✅ ${symbol} をウォッチリストに追加しました（サーバー起動後）`, 'success');
                        } else {
                            showTemporaryMessage(`❌ ${symbol} の追加に失敗しました`, 'error');
                        }
                    } catch (retryError) {
                        showTemporaryMessage(`❌ ${symbol} の追加でAPI接続エラー（再試行後）`, 'error');
                    }
                } else {
                    showTemporaryMessage(`❌ ${symbol} の追加でAPI接続エラー`, 'error');
                }
            }
        }
        
        async function removeFromWatchlistImmediate(symbol, analysisType) {
            try {
                const response = await fetch('http://127.0.0.1:5001/api/watchlist/remove', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({symbol: symbol, analysis_type: analysisType})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showTemporaryMessage(`🗑️ ${symbol} をウォッチリストから削除しました`, 'success');
                } else {
                    showTemporaryMessage(`❌ ${symbol} の削除に失敗しました`, 'error');
                }
            } catch (error) {
                console.error('API呼び出しエラー:', error);
                
                // API接続エラーの場合、サーバー起動を試行
                showTemporaryMessage('⚠️ APIサーバーに接続できません。自動起動を試行中...', 'warning');
                
                const serverStarted = await startApiServerIfNeeded();
                
                if (serverStarted) {
                    // サーバー起動後に再試行
                    try {
                        const retryResponse = await fetch('http://127.0.0.1:5001/api/watchlist/remove', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({symbol: symbol, analysis_type: analysisType})
                        });
                        
                        const retryData = await retryResponse.json();
                        
                        if (retryData.success) {
                            showTemporaryMessage(`🗑️ ${symbol} をウォッチリストから削除しました（サーバー起動後）`, 'success');
                        } else {
                            showTemporaryMessage(`❌ ${symbol} の削除に失敗しました`, 'error');
                        }
                    } catch (retryError) {
                        showTemporaryMessage(`❌ ${symbol} の削除でAPI接続エラー（再試行後）`, 'error');
                    }
                } else {
                    showTemporaryMessage(`❌ ${symbol} の削除でAPI接続エラー`, 'error');
                }
            }
        }
        
        function selectAllWatchlist() {
            const checkboxes = document.querySelectorAll('.watchlist-checkbox');
            checkboxes.forEach(checkbox => {
                if (!checkbox.checked) {
                    checkbox.checked = true;
                    handleWatchlistChange(checkbox);
                }
            });
        }
        
        function deselectAllWatchlist() {
            const checkboxes = document.querySelectorAll('.watchlist-checkbox');
            checkboxes.forEach(checkbox => {
                if (checkbox.checked) {
                    checkbox.checked = false;
                    handleWatchlistChange(checkbox);
                }
            });
        }
        
        // ページ読み込み時に既存のウォッチリスト状態をチェック
        document.addEventListener('DOMContentLoaded', function() {
            checkExistingWatchlistStates();
        });
        
        function checkExistingWatchlistStates() {
            const analysisTypes = ['rsi35_below', 'top_stocks', 'top_score_stocks'];
            
            analysisTypes.forEach(analysisType => {
                fetch(`http://127.0.0.1:5001/api/watchlist?analysis_type=${analysisType}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.data) {
                        const watchedSymbols = data.data.map(item => item.symbol);
                        
                        document.querySelectorAll(`input[data-analysis-type="${analysisType}"]`).forEach(checkbox => {
                            const symbol = checkbox.dataset.symbol;
                            if (watchedSymbols.includes(symbol)) {
                                checkbox.checked = true;
                            }
                        });
                    }
                })
                .catch(error => {
                    console.log('⚠️  ウォッチリスト状態の確認に失敗:', error);
                    console.log('💡 APIサーバーが起動していることを確認してください');
                });
            });
        }
    </script>
    """

def add_simple_watchlist_javascript() -> str:
    """シンプルなウォッチリスト機能用のJavaScript（超堅牢版）"""
    return """
    <script>
        // チェックボックスの変更を処理（即座にウォッチリストに追加/削除、連動機能付き）
        function toggleWatchlistImmediate(checkbox) {
            try {
                const symbol = checkbox.dataset.symbol;
                const analysisType = checkbox.dataset.analysisType;
                
                // メタデータの安全な解析
                let metadata = {};
                try {
                    if (checkbox.dataset.metadata) {
                        metadata = JSON.parse(checkbox.dataset.metadata);
                    }
                } catch (parseError) {
                    console.warn(`⚠️ メタデータ解析エラー for ${symbol}:`, parseError);
                    console.warn(`⚠️ 問題のあるメタデータ:`, checkbox.dataset.metadata);
                    metadata = { symbol: symbol, price: 0, score: 0 }; // フォールバック
                }
                
                console.log(`🔄 チェックボックス変更: ${symbol}, ${analysisType}, checked: ${checkbox.checked}`);
                
                // 同じ銘柄の他のチェックボックスも同期
                syncCheckboxesForSymbol(symbol, checkbox.checked);
                
                if (checkbox.checked) {
                    // ウォッチリストに追加
                    addToWatchlistImmediate([{
                        symbol: symbol,
                        analysisType: analysisType,
                        metadata: metadata
                    }]);
                } else {
                    // ウォッチリストから削除
                    removeFromWatchlistImmediate(symbol, analysisType);
                }
            } catch (error) {
                console.error('❌ チェックボックス処理エラー:', error);
                showTemporaryMessage('❌ チェックボックス処理でエラーが発生しました', 'error');
            }
        }
        
        // 同じ銘柄のすべてのチェックボックスを同期
        function syncCheckboxesForSymbol(symbol, checked) {
            document.querySelectorAll(`input[type="checkbox"][data-symbol="${symbol}"]`).forEach(cb => {
                cb.checked = checked;
            });
            console.log(`🔗 ${symbol} のチェックボックスを同期: ${checked}`);
        }
        
        // ウォッチリストに即座に追加（改良版）
        function addToWatchlistImmediate(stocksArray) {
            console.log('📝 ウォッチリストに追加開始:', stocksArray);
            
            // API通信開始を表示
            showTemporaryMessage(`⏳ ${stocksArray[0].symbol} を追加中...`, 'info');
            
            // 実際のAPI呼び出し
            fetch('http://127.0.0.1:5001/api/watchlist/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(stocksArray)
            })
            .then(response => {
                console.log('📡 API応答受信:', response.status, response.statusText);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.text().then(text => {
                    try {
                        return JSON.parse(text);
                    } catch (parseError) {
                        console.error('❌ 追加API - JSON解析エラー:', parseError);
                        console.error('❌ 問題のあるレスポンス:', text);
                        throw new Error(`Response parsing failed: ${parseError.message}`);
                    }
                });
            })
            .then(data => {
                console.log('📊 API応答データ:', data);
                if (data && data.success) {
                    console.log(`✅ ${stocksArray[0].symbol} をウォッチリストに追加しました`);
                    showTemporaryMessage(`➕ ${stocksArray[0].symbol} をウォッチリストに追加`, 'success');
                } else {
                    const errorMsg = data && data.error ? data.error : '不明なエラー';
                    console.error('❌ ウォッチリストへの追加に失敗:', errorMsg);
                    showTemporaryMessage(`❌ ${stocksArray[0].symbol} の追加に失敗: ${errorMsg}`, 'error');
                    // エラー時はチェックボックスを元に戻す
                    syncCheckboxesForSymbol(stocksArray[0].symbol, false);
                }
            })
            .catch(error => {
                console.error('🚫 API呼び出しエラー:', error);
                showTemporaryMessage(`❌ API接続エラー - サーバーを起動してください: python start_watchlist_api.py`, 'error');
                // エラー時はチェックボックスを元に戻す
                syncCheckboxesForSymbol(stocksArray[0].symbol, false);
            });
        }
        
        // ウォッチリストから即座に削除（改良版）
        function removeFromWatchlistImmediate(symbol, analysisType) {
            console.log('🗑️ ウォッチリストから削除開始:', symbol, analysisType);
            
            // API通信開始を表示
            showTemporaryMessage(`⏳ ${symbol} を削除中...`, 'info');
            
            // 実際のAPI呼び出し
            fetch('http://127.0.0.1:5001/api/watchlist/remove', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({symbol: symbol, analysis_type: analysisType})
            })
            .then(response => {
                console.log('📡 API応答受信:', response.status, response.statusText);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.text().then(text => {
                    try {
                        return JSON.parse(text);
                    } catch (parseError) {
                        console.error('❌ 削除API - JSON解析エラー:', parseError);
                        console.error('❌ 問題のあるレスポンス:', text);
                        throw new Error(`Response parsing failed: ${parseError.message}`);
                    }
                });
            })
            .then(data => {
                console.log('📊 API応答データ:', data);
                if (data && data.success) {
                    console.log(`✅ ${symbol} をウォッチリストから削除しました`);
                    showTemporaryMessage(`➖ ${symbol} をウォッチリストから削除`, 'success');
                } else {
                    const errorMsg = data && data.error ? data.error : '不明なエラー';
                    console.error('❌ ウォッチリストからの削除に失敗:', errorMsg);
                    showTemporaryMessage(`❌ ${symbol} の削除に失敗: ${errorMsg}`, 'error');
                    // エラー時はチェックボックスを元に戻す
                    syncCheckboxesForSymbol(symbol, true);
                }
            })
            .catch(error => {
                console.error('🚫 API呼び出しエラー:', error);
                showTemporaryMessage(`❌ API接続エラー - サーバーを起動してください: python start_watchlist_api.py`, 'error');
                // エラー時はチェックボックスを元に戻す
                syncCheckboxesForSymbol(symbol, true);
            });
        }
        
        // 一時的なメッセージを表示（改良版・表示時間カスタマイズ対応）
        function showTemporaryMessage(message, type = 'success', duration = null) {
            // 既存のメッセージを削除
            const existingMessage = document.querySelector('.watchlist-message');
            if (existingMessage) {
                existingMessage.remove();
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.className = 'watchlist-message';
            messageDiv.textContent = message;
            messageDiv.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 16px;
                border-radius: 6px;
                color: white;
                font-weight: bold;
                z-index: 10000;
                transition: all 0.3s ease;
                max-width: 400px;
                font-size: 14px;
                ${type === 'success' ? 'background-color: #28a745;' : 
                  type === 'error' ? 'background-color: #dc3545;' : 
                  type === 'info' ? 'background-color: #17a2b8;' :
                  'background-color: #ffc107; color: #212529;'}
            `;
            
            document.body.appendChild(messageDiv);
            
            // 表示時間を決定（カスタム時間 > タイプ別デフォルト）
            const displayTime = duration !== null ? duration : (type === 'info' ? 1000 : 3000);
            setTimeout(() => {
                if (messageDiv.parentElement) {
                    messageDiv.remove();
                }
            }, displayTime);
        }
        
        // ページ読み込み時に既存のウォッチリスト状態を確認（改良版）
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 ページ読み込み完了 - ウォッチリスト初期化開始');
            
            // 少し遅延してから初期化（DOMの完全な準備を待つ）
            setTimeout(() => {
                console.log('⏰ DOM安定化後にウォッチリスト初期化を実行');
                checkExistingWatchlistStates();
            }, 500);
        });
        
        // 既存のウォッチリスト状態を確認（完全デバッグ版）
        function checkExistingWatchlistStates() {
            console.log('🔍 ウォッチリスト状態確認を開始');
            console.log('🌐 現在のURL:', window.location.href);
            console.log('📡 APIエンドポイント: http://127.0.0.1:5001/api/watchlist');
            
            // file://プロトコルチェック
            if (window.location.protocol === 'file:') {
                console.info('📁 ファイルプロトコルで開かれています');
                console.info('💡 ヒント: ウォッチリスト機能を使用する場合は http://127.0.0.1:8080/graphs/top_stocks_analysis.html でアクセスしてください');
                showTemporaryMessage('💡 ヒント: ウォッチリスト機能を使用する場合は http://127.0.0.1:8080 でアクセスしてください', 'info', 4000);
                return; // API接続をスキップ
            }
            
            // API接続開始の表示
            showTemporaryMessage('🔄 ウォッチリストを読み込み中...', 'info', 1000);
            
            // まずヘルスチェックを実行
            checkAPIHealth().then(isHealthy => {
                if (!isHealthy) {
                    console.error('⚠️ APIサーバーのヘルスチェックに失敗');
                    showTemporaryMessage('⚠️ ウォッチリストAPIが応答しません。サーバーを起動してください: python start_watchlist_api.py', 'error', 8000);
                    return;
                }
                
                console.log('✅ APIサーバーのヘルスチェック成功');
                // ヘルスチェック成功後、軽量版ウォッチリスト取得を試行
                loadWatchlistWithRetry();
            });
        }
        
        // APIヘルスチェック（軽量・高速）
        function checkAPIHealth() {
            console.log('🏥 APIヘルスチェック開始');
            
            return fetch('http://127.0.0.1:5001/api/health', {
                method: 'GET',
                timeout: 5000 // 5秒タイムアウト
            })
            .then(response => {
                console.log('🏥 ヘルスチェック応答:', response.status);
                return response.ok;
            })
            .catch(error => {
                console.error('🏥 ヘルスチェック失敗:', error);
                return false;
            });
        }
        
        // ウォッチリスト読み込み（リトライ機能付き・改良版）
        function loadWatchlistWithRetry(retryCount = 0, maxRetries = 3) {
            // より段階的なタイムアウト設定: 10秒→20秒→30秒→40秒
            const timeouts = [10000, 20000, 30000, 40000];
            const timeout = timeouts[Math.min(retryCount, timeouts.length - 1)];
            console.log(`🔄 ウォッチリスト読み込み試行 ${retryCount + 1}/${maxRetries + 1} (タイムアウト: ${timeout}ms)`);
            
            // 軽量版ウォッチリスト取得APIを使用
            const fetchWithTimeout = (url, options = {}, timeoutMs = timeout) => {
                console.log(`📡 API呼び出し開始: ${url} (タイムアウト: ${timeoutMs}ms)`);
                
                const timeoutPromise = new Promise((_, reject) => {
                    setTimeout(() => {
                        console.error(`⏰ タイムアウト発生: ${timeoutMs}ms経過`);
                        reject(new Error(`Request timeout after ${timeoutMs}ms`));
                    }, timeoutMs);
                });
                
                const fetchPromise = fetch(url, options).then(response => {
                    console.log(`📡 レスポンス受信完了:`, {
                        status: response.status,
                        statusText: response.statusText,
                        ok: response.ok,
                        url: response.url
                    });
                    return response;
                });
                
                return Promise.race([fetchPromise, timeoutPromise]);
            };
            
            // 基本的なウォッチリスト取得（軽量版）
            fetchWithTimeout('http://127.0.0.1:5001/api/watchlist?lightweight=true', {}, timeout)
            .then(response => {
                console.log('📡 API応答詳細:', {
                    status: response.status,
                    statusText: response.statusText,
                    ok: response.ok,
                    headers: [...response.headers.entries()]
                });
                
                if (!response.ok) {
                    const errorMsg = `API応答エラー: ${response.status} ${response.statusText}`;
                    console.error('❌', errorMsg);
                    throw new Error(errorMsg);
                }
                
                console.log('📤 JSONデータ解析開始...');
                // レスポンステキストを安全に取得してからJSONパース
                return response.text().then(text => {
                    console.log('📤 生テキスト受信:', text.substring(0, 200) + (text.length > 200 ? '...' : ''));
                    
                    if (!text || text.trim() === '') {
                        throw new Error('Empty response received');
                    }
                    
                    try {
                        const parsed = JSON.parse(text);
                        console.log('✅ JSON解析成功');
                        return parsed;
                    } catch (parseError) {
                        console.error('❌ JSON解析エラー:', parseError);
                        console.error('❌ 問題のあるテキスト:', text);
                        throw new Error(`JSON parse error: ${parseError.message}`);
                    }
                });
            })
            .then(data => {
                console.log('📊 受信データ詳細:', {
                    dataType: typeof data,
                    hasSuccess: data && typeof data.success !== 'undefined',
                    success: data ? data.success : 'undefined',
                    hasData: data && typeof data.data !== 'undefined',
                    dataType_inner: data && data.data ? typeof data.data : 'undefined',
                    isArray: data && Array.isArray(data.data),
                    count: data ? data.count : 'undefined',
                    dataLength: data && data.data ? data.data.length : 'undefined'
                });
                
                // 詳細ログ（デバッグ用）
                console.log('📊 RAWデータ全体:', JSON.stringify(data, null, 2));
                
                // データ形式チェック（より詳細）
                if (!data || typeof data !== 'object') {
                    const errorMsg = `データがオブジェクトではありません: ${typeof data}`;
                    console.error('❌', errorMsg);
                    throw new Error(errorMsg);
                }
                
                // success フィールドの確認
                if (typeof data.success === 'undefined') {
                    console.warn('⚠️ レスポンスに success フィールドがありません。フォールバック処理を実行');
                    // フォールバック: data が直接配列の場合
                    if (Array.isArray(data)) {
                        console.log('🔄 data自体が配列形式のため、直接処理します');
                        data = { success: true, data: data };
                    } else {
                        throw new Error('Response missing success field and not an array');
                    }
                }
                
                if (!data.success) {
                    const errorMsg = `APIが失敗を返しました: ${JSON.stringify(data)}`;
                    console.error('❌', errorMsg);
                    throw new Error(errorMsg);
                }
                
                // data.data の確認
                if (!data.data) {
                    console.warn('⚠️ data.data が存在しません。空のウォッチリストとして処理');
                    data.data = [];
                } else if (!Array.isArray(data.data)) {
                    const errorMsg = `data.data が配列ではありません: ${typeof data.data}`;
                    console.error('❌', errorMsg);
                    console.error('❌ 実際の data.data:', data.data);
                    throw new Error(errorMsg);
                }
                
                // ウォッチリスト銘柄を抽出
                const watchedSymbols = new Set();
                data.data.forEach((item, index) => {
                    try {
                        console.log(`📋 アイテム ${index}:`, {
                            item: item,
                            symbol: item ? item.symbol : 'undefined',
                            company_name: item ? item.company_name : 'undefined',
                            analysis_type: item ? item.analysis_type : 'undefined'
                        });
                        
                        if (item && item.symbol && typeof item.symbol === 'string' && item.symbol.trim()) {
                            watchedSymbols.add(item.symbol.trim());
                        } else {
                            console.warn(`⚠️ 無効なシンボル at index ${index}:`, item);
                        }
                    } catch (itemError) {
                        console.error(`❌ アイテム処理エラー at index ${index}:`, itemError);
                    }
                });
                
                console.log('📋 抽出されたウォッチリスト銘柄:', Array.from(watchedSymbols));
                
                // チェックボックスを更新
                updateCheckboxStates(watchedSymbols);
                
                // 成功メッセージ
                if (watchedSymbols.size > 0) {
                    showTemporaryMessage(`📋 ウォッチリスト読み込み完了: ${watchedSymbols.size}銘柄`, 'success', 3000);
                } else {
                    console.log('📋 ウォッチリストは空です');
                    showTemporaryMessage('📋 ウォッチリストは空です', 'info', 2000);
                }
            })
            .catch(error => {
                console.error(`⚠️ ウォッチリスト読み込み失敗 (試行 ${retryCount + 1}):`, error);
                console.error(`🔧 例外詳細:`, {
                    name: error.name,
                    message: error.message,
                    stack: error.stack,
                    toString: error.toString()
                });
                
                // リトライ判定
                if (retryCount < maxRetries) {
                    const retryDelay = Math.min(2000 + (retryCount * 1000), 5000); // 2秒→3秒→4秒→5秒
                    console.log(`🔄 ${retryDelay}ms後にリトライします...`);
                    showTemporaryMessage(`⏳ 読み込み中... (${retryCount + 1}/${maxRetries + 1}回目) ${error.message.includes('timeout') ? '[タイムアウト]' : '[エラー]'}`, 'info', retryDelay);
                    setTimeout(() => {
                        loadWatchlistWithRetry(retryCount + 1, maxRetries);
                    }, retryDelay);
                } else {
                    console.error('❌ 最大リトライ回数に達しました');
                    handleWatchlistLoadError(error);
                }
            });
        }
        
        // チェックボックス状態更新（分離された関数）
        function updateCheckboxStates(watchedSymbols) {
            const checkboxes = document.querySelectorAll('input[type="checkbox"][data-symbol]');
            console.log(`🔍 ページ内のチェックボックス検索結果: ${checkboxes.length}個`);
            
            if (checkboxes.length === 0) {
                console.warn('⚠️ チェックボックスが見つかりません。DOM構造を確認してください。');
                const tableRows = document.querySelectorAll('table tr');
                console.log(`🔍 テーブル行数: ${tableRows.length}`);
                console.log('🔍 最初の数行の構造:', Array.from(tableRows).slice(0, 3).map(row => row.innerHTML));
            }
            
            let updatedCount = 0;
            let skippedCount = 0;
            
            checkboxes.forEach((checkbox, index) => {
                const symbol = checkbox.dataset.symbol;
                console.log(`🔍 チェックボックス ${index}:`, {
                    symbol: symbol,
                    currentChecked: checkbox.checked,
                    shouldBeChecked: watchedSymbols.has(symbol)
                });
                
                if (symbol && typeof symbol === 'string') {
                    const shouldBeChecked = watchedSymbols.has(symbol);
                    
                    if (checkbox.checked !== shouldBeChecked) {
                        checkbox.checked = shouldBeChecked;
                        updatedCount++;
                        console.log(`🔄 ${symbol} 更新: ${!shouldBeChecked} → ${shouldBeChecked}`);
                    }
                } else {
                    skippedCount++;
                    console.warn(`⚠️ 無効なシンボル (チェックボックス ${index}):`, symbol);
                }
            });
            
            console.log(`✅ チェックボックス更新完了:`, {
                totalCheckboxes: checkboxes.length,
                updatedCount: updatedCount,
                skippedCount: skippedCount,
                watchlistSize: watchedSymbols.size
            });
        }
        
        // ウォッチリスト読み込みエラーハンドリング（改良版）
        function handleWatchlistLoadError(error) {
            console.error('🔧 例外詳細:', {
                name: error.name,
                message: error.message,
                stack: error.stack,
                toString: error.toString()
            });
            
            // 詳細なエラー分類とユーザーフレンドリーなメッセージ
            let errorMessage = '';
            let errorType = 'error';
            let helpMessage = '';
            
            if (error.message.includes('timeout') || error.message.includes('Request timeout')) {
                errorMessage = '⚠️ ウォッチリストAPIがタイムアウトしました';
                errorType = 'warning';
                helpMessage = '💡 データベースが重い可能性があります。朝夕の更新時間帯は応答が遅くなることがあります';
            } else if (error.message.includes('Failed to fetch') || error.name === 'TypeError' || error.message.includes('NetworkError')) {
                errorMessage = '⚠️ ウォッチリストAPIサーバーに接続できません';
                errorType = 'error';
                helpMessage = '🚀 解決方法: ターミナルで「python start_watchlist_api.py」を実行してサーバーを起動してください';
            } else if (error.message.includes('JSON parse error') || error.message.includes('SyntaxError')) {
                errorMessage = '⚠️ APIレスポンスが正しいJSON形式ではありません';
                errorType = 'warning';
                helpMessage = '🔧 APIサーバーがHTMLエラーページやプレーンテキストを返している可能性があります';
            } else if (error.message.includes('not an array') || error.message.includes('Invalid data format')) {
                errorMessage = '⚠️ APIレスポンスのデータ構造が予期しない形式です';
                errorType = 'warning';
                helpMessage = '🔄 APIサーバーのバージョンが古い可能性があります。サーバーを再起動してください';
            } else if (error.message.includes('API応答エラー')) {
                errorMessage = `⚠️ ${error.message}`;
                errorType = 'warning';
                helpMessage = '📡 APIサーバーがHTTPエラーステータスを返しました';
            } else if (error.message.includes('success: false') || error.message.includes('API returned success: false')) {
                errorMessage = '⚠️ APIサーバーが処理エラーを報告しました';
                errorType = 'warning';
                helpMessage = '🔍 データベース接続やクエリにエラーがある可能性があります';
            } else if (error.message.includes('Empty response received')) {
                errorMessage = '⚠️ APIサーバーから空のレスポンスを受信しました';
                errorType = 'warning';
                helpMessage = '🔄 APIサーバーが過負荷状態の可能性があります';
            } else {
                errorMessage = `⚠️ 予期しないエラー: ${error.message}`;
                errorType = 'error';
                helpMessage = '🔧 詳細はブラウザのコンソールログを確認してください';
            }
            
            console.log(`📢 ユーザーへのエラーメッセージ: ${errorMessage}`);
            console.log(`💡 ヘルプメッセージ: ${helpMessage}`);
            
            // エラーメッセージを表示
            showTemporaryMessage(errorMessage, errorType, 8000);
            
            // ヘルプメッセージを少し遅れて表示
            setTimeout(() => {
                showTemporaryMessage(helpMessage, 'info', 10000);
            }, 1000);
            
            // さらに遅れて一般的な解決策を表示
            setTimeout(() => {
                showTemporaryMessage('🔄 ヒント: ページをリロード(⌘+R / Ctrl+R)して再試行してください', 'info', 6000);
            }, 3000);
        }
    </script>
    """


def add_simple_watchlist_css() -> str:
    """シンプルなウォッチリスト機能用のCSS"""
    return """
    <style>
        /* シンプルなウォッチリストチェックボックス */
        .simple-watchlist-checkbox {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            margin-left: 10px;
            padding: 5px 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            transition: all 0.2s ease;
        }
        
        .simple-watchlist-checkbox:hover {
            background-color: #e9ecef;
            border-color: #3498db;
        }
        
        .simple-watchlist-checkbox input[type="checkbox"] {
            margin: 0;
            cursor: pointer;
        }
        
        .simple-watchlist-checkbox label {
            margin: 0;
            font-size: 12px;
            color: #495057;
            cursor: pointer;
            user-select: none;
        }
        
        .simple-watchlist-checkbox input[type="checkbox"]:checked + label {
            color: #28a745;
            font-weight: bold;
        }
        
        /* サマリーテーブル内のチェックボックス */
        .summary-table .simple-watchlist-checkbox {
            justify-content: center;
            background-color: transparent;
            border: none;
            padding: 2px;
        }
        
        .summary-table .simple-watchlist-checkbox:hover {
            background-color: rgba(52, 152, 219, 0.1);
        }
    </style>
    """


def serialize_metadata_safely(metadata):
    """メタデータを安全にシリアライズ（日付オブジェクトを文字列に変換）"""
    if metadata is None:
        return {}
    
    def convert_dates(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: convert_dates(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_dates(item) for item in obj]
        else:
            return obj
    
    return convert_dates(metadata)


def generate_simple_watchlist_checkbox(symbol, analysis_type, metadata=None):
    """シンプルなウォッチリスト用チェックボックスHTMLを生成"""
    # メタデータを安全にシリアライズ
    safe_metadata = serialize_metadata_safely(metadata)
    metadata_json = html.escape(json.dumps(safe_metadata))
    
    checkbox_html = f'''
        <div class="simple-watchlist-checkbox">
            <input type="checkbox" 
                   id="watch_{symbol}" 
                   data-symbol="{symbol}" 
                   data-analysis-type="{analysis_type}"
                   data-metadata='{metadata_json}'
                   onchange="toggleWatchlistImmediate(this)">
            <label for="watch_{symbol}">Watch</label>
        </div>
    '''
    return checkbox_html

def get_market_global_ranking(engine, symbol: str, market_type: str, target_date: str) -> dict:
    """
    指定銘柄の市場別全銘柄中でのランキングを取得
    
    Args:
        engine: データベースエンジン
        symbol: 銘柄コード
        market_type: 'US' または 'JP'
        target_date: 対象日
        
    Returns:
        {"rank": 順位, "total_stocks": 総銘柄数}
    """
    # 市場フィルター（既存のget_top_scored_stocks_by_market関数と同じロジック）
    if market_type == 'JP':
        market_filter = "(symbol LIKE '%.T' OR (symbol ~ '^[0-9]{4}$'))"
    else:
        market_filter = "(symbol NOT LIKE '%.T' AND NOT (symbol ~ '^[0-9]{4}$'))"
    
    query = text(f"""
    WITH ranked_scores AS (
        SELECT 
            symbol,
            ROW_NUMBER() OVER (ORDER BY total_score DESC) as rank
        FROM backtest_results.daily_scores
        WHERE date = :target_date AND {market_filter}
    ),
    total_count AS (
        SELECT COUNT(*) as total_stocks
        FROM backtest_results.daily_scores
        WHERE date = :target_date AND {market_filter}
    )
    SELECT 
        COALESCE(r.rank, 0) as rank,
        t.total_stocks
    FROM total_count t
    LEFT JOIN ranked_scores r ON r.symbol = :symbol
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {
                "target_date": target_date, 
                "symbol": symbol
            }).fetchone()
            
            return {
                "rank": result.rank if result else 0,
                "total_stocks": result.total_stocks if result else 0
            }
    except Exception as e:
        print(f"グローバルランキング取得エラー {symbol}: {e}")
        return {"rank": 0, "total_stocks": 0}


def get_top_scored_stocks_by_market(engine, target_date: str = None, top_n: int = 5, market_type: str = 'US') -> pd.DataFrame:
    """
    指定日のスコア上位銘柄を市場別に取得
    
    Args:
        engine: データベースエンジン
        target_date: 対象日（Noneの場合は市場別の最新日）
        top_n: 上位何銘柄を取得するか
        market_type: 'US' (米国株) または 'JP' (日本株)
        
    Returns:
        上位銘柄のスコアデータ
    """
    # 市場フィルターの条件
    if market_type == 'JP':
        # 日本株: .T で終わるまたは4桁数字のみ
        market_filter = "(symbol LIKE '%.T' OR (symbol ~ '^[0-9]{4}$'))"
        market_name = "日本株"
    else:
        # 米国株: .T で終わらない、かつ4桁数字のみでない
        market_filter = "(symbol NOT LIKE '%.T' AND NOT (symbol ~ '^[0-9]{4}$'))"
        market_name = "米国株"
    
    if target_date is None:
        # 市場別の最新日を取得
        date_query = text(f"""
        SELECT MAX(date) as max_date 
        FROM backtest_results.daily_scores 
        WHERE {market_filter}
        """)
        with engine.connect() as conn:
            result = conn.execute(date_query).fetchone()
            if result.max_date is None:
                print(f"  {market_name}: データが見つかりません")
                return pd.DataFrame()
            target_date = result.max_date.strftime('%Y-%m-%d')
            print(f"  {market_name}の最新データ日付: {target_date}")
    
    # 上位銘柄のスコアを取得
    query = text(f"""
    SELECT 
        symbol,
        date,
        total_score,
        value_score,
        growth_score,
        quality_score,
        momentum_score,
        macro_sector_score,
        per_score,
        fcf_yield_score,
        ev_ebitda_score,
        eps_cagr_score,
        revenue_cagr_score,
        growth_consistency_score,
        roic_score,
        roe_score,
        debt_equity_score,
        altman_z_score,
        piotroski_f_score,
        cash_conversion_score,
        golden_cross_score,
        rsi_score,
        macd_hist_score,
        vol_adj_momentum_score,
        relative_strength_score,
        tail_wind_score,
        sector_rotation_score,
        '{market_type}' as market_type,
        'all' as ranking_type
    FROM backtest_results.daily_scores
    WHERE date = :target_date
      AND {market_filter}
    ORDER BY total_score DESC
    LIMIT :top_n
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"target_date": target_date, "top_n": top_n})
        print(f"  {market_name}スコア上位銘柄取得: {len(df)}件 (日付: {target_date})")
        return df
    except Exception as e:
        print(f"{market_name}スコアデータ取得エラー: {e}")
        return pd.DataFrame()


def get_top_scored_stocks_by_market_filtered(engine, target_date: str = None, top_n: int = 5, market_type: str = 'US') -> pd.DataFrame:
    """
    指定日のスコア上位銘柄を市場別に取得（フィルター適用版）
    
    Args:
        engine: データベースエンジン
        target_date: 対象日（Noneの場合は市場別の最新日）
        top_n: 上位何銘柄を取得するか
        market_type: 'US' (米国株) または 'JP' (日本株)
        
    Returns:
        上位銘柄のスコアデータ（フィルター済み）
    """
    # 市場フィルターの条件
    if market_type == 'JP':
        # 日本株: .T で終わるまたは4桁数字のみ
        market_filter = "(symbol LIKE '%.T' OR (symbol ~ '^[0-9]{4}$'))"
        market_name = "日本株"
    else:
        # 米国株: .T で終わらない、かつ4桁数字のみでない
        market_filter = "(symbol NOT LIKE '%.T' AND NOT (symbol ~ '^[0-9]{4}$'))"
        market_name = "米国株"
    
    if target_date is None:
        # 市場別の最新日を取得
        date_query = text(f"""
        SELECT MAX(date) as max_date 
        FROM backtest_results.daily_scores 
        WHERE {market_filter}
          AND is_value_trap_filtered = FALSE 
          AND is_quality_growth_filtered = FALSE
        """)
        with engine.connect() as conn:
            result = conn.execute(date_query).fetchone()
            if result.max_date is None:
                print(f"  {market_name}(フィルター済み): データが見つかりません")
                return pd.DataFrame()
            target_date = result.max_date.strftime('%Y-%m-%d')
            print(f"  {market_name}(フィルター済み)の最新データ日付: {target_date}")
    
    # 上位銘柄のスコアを取得（フィルター適用）
    query = text(f"""
    SELECT 
        symbol,
        date,
        total_score,
        value_score,
        growth_score,
        quality_score,
        momentum_score,
        macro_sector_score,
        per_score,
        fcf_yield_score,
        ev_ebitda_score,
        eps_cagr_score,
        revenue_cagr_score,
        growth_consistency_score,
        roic_score,
        roe_score,
        debt_equity_score,
        altman_z_score,
        piotroski_f_score,
        cash_conversion_score,
        golden_cross_score,
        rsi_score,
        macd_hist_score,
        vol_adj_momentum_score,
        relative_strength_score,
        tail_wind_score,
        sector_rotation_score,
        '{market_type}' as market_type,
        'filtered' as ranking_type
    FROM backtest_results.daily_scores
    WHERE date = :target_date
      AND {market_filter}
      AND is_value_trap_filtered = FALSE 
      AND is_quality_growth_filtered = FALSE
    ORDER BY total_score DESC
    LIMIT :top_n
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"target_date": target_date, "top_n": top_n})
        print(f"  {market_name}(フィルター済み)スコア上位銘柄取得: {len(df)}件 (日付: {target_date})")
        return df
    except Exception as e:
        print(f"{market_name}(フィルター済み)スコアデータ取得エラー: {e}")
        return pd.DataFrame()


def get_top_scored_stocks(engine, target_date: str = None, top_n: int = 10) -> pd.DataFrame:
    """
    指定日のスコア上位銘柄を取得
    
    Args:
        engine: データベースエンジン
        target_date: 対象日（Noneの場合は最新日）
        top_n: 上位何銘柄を取得するか
        
    Returns:
        上位銘柄のスコアデータ
    """
    if target_date is None:
        # 最新日を取得
        date_query = text("SELECT MAX(date) as max_date FROM backtest_results.daily_scores")
        with engine.connect() as conn:
            result = conn.execute(date_query).fetchone()
            target_date = result.max_date.strftime('%Y-%m-%d')
    
    # 上位銘柄のスコアを取得
    query = text("""
    SELECT 
        symbol,
        date,
        total_score,
        value_score,
        growth_score,
        quality_score,
        momentum_score,
        macro_sector_score,
        per_score,
        fcf_yield_score,
        ev_ebitda_score,
        eps_cagr_score,
        revenue_cagr_score,
        growth_consistency_score,
        roic_score,
        roe_score,
        debt_equity_score,
        altman_z_score,
        piotroski_f_score,
        cash_conversion_score,
        golden_cross_score,
        rsi_score,
        macd_hist_score,
        vol_adj_momentum_score,
        relative_strength_score,
        tail_wind_score,
        sector_rotation_score
    FROM backtest_results.daily_scores
    WHERE date = :target_date
    ORDER BY total_score DESC
    LIMIT :top_n
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"target_date": target_date, "top_n": top_n})
        return df
    except Exception as e:
        print(f"スコアデータ取得エラー: {e}")
        return pd.DataFrame()


def get_stock_fundamental_data(engine, symbol: str, years_back: int = 5) -> pd.DataFrame:
    """
    銘柄の財務データの時系列を取得
    
    Args:
        engine: データベースエンジン
        symbol: 銘柄コード
        years_back: 何年分のデータを取得するか
        
    Returns:
        財務データの時系列
    """
    start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
    
    query = text("""
    SELECT 
        date,
        per,
        pbr,
        roe,
        roic,
        fcf_yield,
        debt_to_equity,
        eps_cagr_3y,
        eps_cagr_5y,
        revenue_cagr_3y,
        revenue_cagr_5y,
        cfo_to_net_income,
        market_cap
    FROM backtest_results.vw_daily_master
    WHERE symbol = :symbol
      AND date >= :start_date
    ORDER BY date
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"symbol": symbol, "start_date": start_date})
        return df
    except Exception as e:
        print(f"財務データ取得エラー {symbol}: {e}")
        return pd.DataFrame()


def get_stock_technical_data(engine, symbol: str, days_back: int = 252) -> pd.DataFrame:
    """
    銘柄のテクニカルデータを取得
    
    Args:
        engine: データベースエンジン
        symbol: 銘柄コード
        days_back: 何日分のデータを取得するか
        
    Returns:
        テクニカルデータ
    """
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    query = text("""
    SELECT
        p.date,
        p.open,
        p.high,
        p.low,
        p.close,
        ti.sma_20,
        ti.sma_40,
        ti.rsi_14,
        ti.macd_hist,
        ti.atr_14,
        p.volume
    FROM fmp_data.daily_prices p
    LEFT JOIN calculated_metrics.technical_indicators ti ON p.symbol = ti.symbol AND p.date = ti.date
    WHERE p.symbol = :symbol
      AND p.date >= :start_date
    ORDER BY p.date
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"symbol": symbol, "start_date": start_date})
        return df
    except Exception as e:
        print(f"テクニカルデータ取得エラー {symbol}: {e}")
        return pd.DataFrame()


def get_stock_weekly_data(engine, symbol: str, weeks_back: int = 52) -> pd.DataFrame:
    """
    銘柄の週足データを取得（デフォルト1年）
    
    Args:
        engine: データベースエンジン
        symbol: 銘柄コード
        weeks_back: 何週分のデータを取得するか（デフォルト52週=1年）
        
    Returns:
        週足データ
    """
    start_date = (datetime.now() - timedelta(weeks=weeks_back)).strftime('%Y-%m-%d')
    
    query = text("""
    SELECT 
        wp.week_start_date as date,
        wp.open,
        wp.high,
        wp.low,
        wp.close,
        wp.volume,
        ti.sma_26w
    FROM calculated_metrics.weekly_prices wp
    LEFT JOIN calculated_metrics.technical_indicators_weekly ti 
        ON wp.symbol = ti.symbol AND wp.week_start_date = ti.week_start_date
    WHERE wp.symbol = :symbol
      AND wp.week_start_date >= :start_date
    ORDER BY wp.week_start_date
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"symbol": symbol, "start_date": start_date})
        return df
    except Exception as e:
        print(f"週足データ取得エラー {symbol}: {e}")
        return pd.DataFrame()


def get_stock_financial_metrics(engine, symbol: str, years_back: int = 5) -> pd.DataFrame:
    """
    銘柄の詳細財務指標を取得
    
    Args:
        engine: データベースエンジン
        symbol: 銘柄コード
        years_back: 何年分のデータを取得するか
        
    Returns:
        詳細財務指標データ
    """
    start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
    
    # TTM収益データ
    income_query = text("""
    SELECT 
        as_of_date as date,
        revenue,
        operating_income,
        net_income,
        CASE WHEN revenue > 0 THEN operating_income / revenue * 100 ELSE NULL END as operating_margin,
        CASE WHEN revenue > 0 THEN net_income / revenue * 100 ELSE NULL END as net_margin
    FROM calculated_metrics.ttm_income_statements
    WHERE symbol = :symbol
      AND as_of_date >= :start_date
    ORDER BY as_of_date
    """)
    
    # バリュエーション指標（PEGを除去）
    valuation_query = text("""
    SELECT 
        as_of_date as date,
        ev_ebitda
    FROM calculated_metrics.composite_valuation_metrics
    WHERE symbol = :symbol
      AND as_of_date >= :start_date
    ORDER BY as_of_date
    """)
    
    # 基本指標
    basic_query = text("""
    SELECT 
        as_of_date as date,
        fcf_yield,
        roe,
        roic
    FROM calculated_metrics.basic_metrics
    WHERE symbol = :symbol
      AND as_of_date >= :start_date
    ORDER BY as_of_date
    """)
    
    # 株式数データ（正しいカラム名を使用）
    shares_query = text("""
    SELECT 
        date,
        float_shares,
        outstanding_shares
    FROM fmp_data.shares
    WHERE symbol = :symbol
      AND date >= :start_date
    ORDER BY date
    """)
    
    # 財務安全性指標（debt_to_equity, current_ratio）をTTM貸借対照表から計算
    financial_safety_query = text("""
    SELECT 
        as_of_date as date,
        CASE 
            WHEN total_stockholders_equity > 0 AND total_debt IS NOT NULL
            THEN total_debt / total_stockholders_equity 
            ELSE NULL 
        END as debt_to_equity,
        CASE 
            WHEN total_current_liabilities > 0 AND total_current_assets IS NOT NULL
            THEN total_current_assets / total_current_liabilities 
            ELSE NULL 
        END as current_ratio
    FROM calculated_metrics.ttm_balance_sheets
    WHERE symbol = :symbol
      AND as_of_date >= :start_date
      AND (total_stockholders_equity IS NOT NULL OR total_current_assets IS NOT NULL)
    ORDER BY as_of_date
    """)
    
    try:
        income_df = pd.read_sql(income_query, engine, params={"symbol": symbol, "start_date": start_date})
        valuation_df = pd.read_sql(valuation_query, engine, params={"symbol": symbol, "start_date": start_date})
        basic_df = pd.read_sql(basic_query, engine, params={"symbol": symbol, "start_date": start_date})
        shares_df = pd.read_sql(shares_query, engine, params={"symbol": symbol, "start_date": start_date})
        financial_safety_df = pd.read_sql(financial_safety_query, engine, params={"symbol": symbol, "start_date": start_date})
        
        # データを統合
        result_df = income_df
        if not valuation_df.empty:
            result_df = pd.merge(result_df, valuation_df, on='date', how='outer')
        if not basic_df.empty:
            result_df = pd.merge(result_df, basic_df, on='date', how='outer')
        if not shares_df.empty:
            result_df = pd.merge(result_df, shares_df, on='date', how='outer')
        if not financial_safety_df.empty:
            result_df = pd.merge(result_df, financial_safety_df, on='date', how='outer')
            
        return result_df.sort_values('date')
    except Exception as e:
        print(f"詳細財務指標取得エラー {symbol}: {e}")
        return pd.DataFrame()


def get_sector_comparison_data(engine, symbol: str, days_back: int = 252) -> pd.DataFrame:
    """
    セクター比較データを取得（通貨別）
    
    Args:
        engine: データベースエンジン
        symbol: 銘柄コード
        days_back: 何日分のデータを取得するか
        
    Returns:
        セクター比較データ
    """
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    # まず銘柄のセクター情報と通貨情報を取得
    sector_query = text("""
    SELECT vm.raw_sector, cp.currency
    FROM backtest_results.vw_daily_master vm
    LEFT JOIN fmp_data.company_profile cp ON vm.symbol = cp.symbol
    WHERE vm.symbol = :symbol 
    AND vm.raw_sector IS NOT NULL
    ORDER BY vm.date DESC
    LIMIT 1
    """)
    
    try:
        with engine.connect() as conn:
            sector_result = conn.execute(sector_query, {"symbol": symbol}).fetchone()
            if not sector_result:
                print(f"Warning: セクター情報が見つかりません - {symbol}")
                return pd.DataFrame()
            
            sector = sector_result[0]
            currency = sector_result[1] if sector_result[1] else 'USD'  # デフォルトUSD
            
            # 日本株の場合はJPYに変換
            if symbol.endswith('.T'):
                currency = 'JPY'
        
        # 通貨に基づいてセクター中央値データを取得し、適切な値で正規化
        if currency == 'JPY':
            # 日本株のセクターデータ取得（有効な値のみ）
            comparison_query = text("""
            SELECT 
                trade_date as date,
                avg_close,
                symbol_count
            FROM calculated_metrics.sector_daily_prices
            WHERE group_name = :sector
              AND trade_date >= :start_date
              AND currency = 'JPY'  -- 日本株のみ
              AND avg_close > 0     -- 有効な値のみ
            ORDER BY trade_date
            """)
        else:
            # 米国株のセクターデータ取得（有効な値のみ）
            comparison_query = text("""
            SELECT 
                trade_date as date,
                avg_close,
                symbol_count
            FROM calculated_metrics.sector_daily_prices
            WHERE group_name = :sector
              AND trade_date >= :start_date
              AND currency = 'USD'  -- 米国株のみ
              AND avg_close > 0     -- 有効な値のみ
            ORDER BY trade_date
            """)
        
        df = pd.read_sql(comparison_query, engine, params={"sector": sector, "start_date": start_date})
        
        # データの品質をチェック
        if not df.empty:
            print(f"Debug: セクターデータ取得 - {len(df)}行, 価格範囲: {df['avg_close'].min():.2f} - {df['avg_close'].max():.2f}")
            if 'symbol_count' in df.columns:
                print(f"Debug: 最新データ - 日付: {df['date'].max()}, 価格: {df['avg_close'].iloc[-1]:.2f}, 銘柄数: {df['symbol_count'].iloc[-1]}")
            else:
                print(f"Debug: 最新データ - 日付: {df['date'].max()}, 価格: {df['avg_close'].iloc[-1]:.2f}")
            print(f"Debug: 最古データ - 日付: {df['date'].min()}, 価格: {df['avg_close'].iloc[0]:.2f}")
            print(f"Debug: セクターデータのサンプル（最新5行）:")
            print(df.tail().to_string())
            
            # データの品質をチェック
            median_price = df['avg_close'].median()
            if median_price > 0:
                # 中央値の1/100から100倍の範囲で制限
                original_count = len(df)
                df = df[
                    (df['avg_close'] >= median_price / 100) &
                    (df['avg_close'] <= median_price * 100)
                ]
                if len(df) != original_count:
                    print(f"Debug: 異常値除外 - {original_count}行 → {len(df)}行")
        else:
            print(f"Warning: セクターデータが見つかりません - セクター: {sector}, 通貨: {currency}")
        
        return df
    except Exception as e:
        print(f"セクター比較データ取得エラー {symbol}: {e}")
        return pd.DataFrame()


def get_stock_basic_info(engine, symbol: str) -> Dict:
    """
    銘柄の基本情報を取得
    
    Args:
        engine: データベースエンジン
        symbol: 銘柄コード
        
    Returns:
        基本情報の辞書
    """
    # まず最新データを取得（追加の財務指標を含む）
    query = text("""
    SELECT 
        vm.symbol,
        vm.raw_industry,
        vm.raw_sector,
        vm.close as current_price,
        vm.market_cap,
        vm.per,
        vm.pbr,
        vm.roe,
        vm.roic,
        cp.company_name,
        bm.debt_to_equity,
        bm.operating_margin,
        bm.gross_margin,
        bm.fcf_yield
    FROM backtest_results.vw_daily_master vm
    LEFT JOIN (
        SELECT DISTINCT ON (symbol) symbol, company_name
        FROM fmp_data.company_profile
        ORDER BY symbol, date DESC
    ) cp ON vm.symbol = cp.symbol
    LEFT JOIN (
        SELECT DISTINCT ON (symbol) 
            symbol, debt_to_equity, operating_margin, gross_margin, fcf_yield
        FROM calculated_metrics.basic_metrics
        WHERE debt_to_equity IS NOT NULL OR operating_margin IS NOT NULL 
           OR gross_margin IS NOT NULL OR fcf_yield IS NOT NULL
        ORDER BY symbol, as_of_date DESC
    ) bm ON vm.symbol = bm.symbol
    WHERE vm.symbol = :symbol
    ORDER BY vm.date DESC
    LIMIT 1
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"symbol": symbol})
        if not df.empty:
            row = df.iloc[0]
            
            # PBR/ROEが欠損している場合、利用可能な最新のPBR/ROEデータを取得（期間制限なし）
            pbr_value = row['pbr']
            roe_value = row['roe']
            
            if pd.isna(pbr_value) or pd.isna(roe_value):
                fallback_query = text("""
                SELECT pbr, roe
                FROM calculated_metrics.basic_metrics
                WHERE symbol = :symbol
                  AND (pbr IS NOT NULL OR roe IS NOT NULL)
                ORDER BY as_of_date DESC
                LIMIT 1
                """)
                
                fallback_df = pd.read_sql(fallback_query, engine, params={"symbol": symbol})
                if not fallback_df.empty:
                    fallback_row = fallback_df.iloc[0]
                    if pd.isna(pbr_value) and not pd.isna(fallback_row['pbr']):
                        pbr_value = fallback_row['pbr']
                    if pd.isna(roe_value) and not pd.isna(fallback_row['roe']):
                        roe_value = fallback_row['roe']
            
            
            return {
                'symbol': row['symbol'],
                'company_name': row['company_name'] or 'N/A',
                'industry': row['raw_industry'] or 'N/A',
                'sector': row['raw_sector'] or 'N/A',
                'current_price': row['current_price'] or 0,
                'market_cap': row['market_cap'] or 0,
                'per': row['per'] or 0,
                'pbr': pbr_value or 0,
                'roe': roe_value or 0,
                'roic': row['roic'] or 0,
                'debt_to_equity': row['debt_to_equity'],
                'operating_margin': row['operating_margin'],
                'net_margin': row['gross_margin'],
                'fcf_yield': row['fcf_yield']
            }
        else:
            return {
                'symbol': symbol,
                'company_name': 'N/A',
                'industry': 'N/A',
                'sector': 'N/A',
                'current_price': 0,
                'market_cap': 0,
                'per': 0,
                'pbr': 0,
                'roe': 0,
                'roic': 0,
                'debt_to_equity': None,
                'operating_margin': None,
                'net_margin': None,
                'fcf_yield': None
            }
    except Exception as e:
        print(f"基本情報取得エラー {symbol}: {e}")
        return {
            'symbol': symbol,
            'company_name': 'N/A',
            'industry': 'N/A',
            'sector': 'N/A',
            'current_price': 0,
            'market_cap': 0,
            'per': 0,
            'pbr': 0,
            'roe': 0,
            'roic': 0,
            'debt_to_equity': None,
            'operating_margin': None,
            'net_margin': None,
            'fcf_yield': None
        }


def analyze_score_components(row: pd.Series) -> Dict[str, str]:
    """
    スコア構成要素を分析して強み・弱みを特定
    
    Args:
        row: スコアデータの行
        
    Returns:
        分析結果の辞書
    """
    analysis = {
        'strengths': [],
        'weaknesses': [],
        'value_analysis': '',
        'growth_analysis': '',
        'quality_analysis': '',
        'momentum_analysis': '',
        'macro_analysis': ''
    }
    
    # Value分析（20点満点）
    value_score = row.get('value_score', 0)
    if value_score >= 15:
        analysis['strengths'].append('割安性')
        analysis['value_analysis'] = f"非常に割安（{value_score:.1f}/20点）"
    elif value_score >= 10:
        analysis['value_analysis'] = f"やや割安（{value_score:.1f}/20点）"
    else:
        analysis['weaknesses'].append('割安性')
        analysis['value_analysis'] = f"割高傾向（{value_score:.1f}/20点）"
    
    # Growth分析（20点満点）
    growth_score = row.get('growth_score', 0)
    if growth_score >= 15:
        analysis['strengths'].append('成長性')
        analysis['growth_analysis'] = f"高成長（{growth_score:.1f}/20点）"
    elif growth_score >= 10:
        analysis['growth_analysis'] = f"安定成長（{growth_score:.1f}/20点）"
    else:
        analysis['weaknesses'].append('成長性')
        analysis['growth_analysis'] = f"成長鈍化（{growth_score:.1f}/20点）"
    
    # Quality分析（20点満点）
    quality_score = row.get('quality_score', 0)
    if quality_score >= 16:
        analysis['strengths'].append('財務品質')
        analysis['quality_analysis'] = f"優良企業（{quality_score:.1f}/20点）"
    elif quality_score >= 12:
        analysis['quality_analysis'] = f"良好な財務（{quality_score:.1f}/20点）"
    else:
        analysis['weaknesses'].append('財務品質')
        analysis['quality_analysis'] = f"財務要改善（{quality_score:.1f}/20点）"
    
    # Momentum分析（20点満点）
    momentum_score = row.get('momentum_score', 0)
    if momentum_score >= 15:
        analysis['strengths'].append('モメンタム')
        analysis['momentum_analysis'] = f"強い上昇勢い（{momentum_score:.1f}/20点）"
    elif momentum_score >= 10:
        analysis['momentum_analysis'] = f"安定推移（{momentum_score:.1f}/20点）"
    else:
        analysis['weaknesses'].append('モメンタム')
        analysis['momentum_analysis'] = f"弱い勢い（{momentum_score:.1f}/20点）"
    
    # Risk分析（10点満点）
    risk_score = row.get('risk_score', 0)
    if risk_score >= 8:
        analysis['strengths'].append('リスク管理')
        analysis['risk_analysis'] = f"低リスク（{risk_score:.1f}/10点）"
    elif risk_score >= 5:
        analysis['risk_analysis'] = f"中程度リスク（{risk_score:.1f}/10点）"
    else:
        analysis['weaknesses'].append('リスク管理')
        analysis['risk_analysis'] = f"高リスク（{risk_score:.1f}/10点）"
    
    return analysis


def generate_investment_recommendation(row: pd.Series, technical_data: pd.DataFrame, basic_info: Dict) -> Dict[str, str]:
    """
    投資判断を生成
    
    Args:
        row: スコアデータの行
        technical_data: テクニカルデータ
        basic_info: 基本情報
        
    Returns:
        投資判断の辞書
    """
    recommendation = {
        'action': '',
        'reasoning': '',
        'risk_level': '',
        'time_horizon': '',
        'entry_strategy': '',
        'exit_strategy': ''
    }
    
    total_score = row.get('total_score', 0)
    value_score = row.get('value_score', 0)
    momentum_score = row.get('momentum_score', 0)
    
    # 最新のテクニカル指標
    if not technical_data.empty:
        latest = technical_data.iloc[-1]
        current_price = latest.get('close', 0)
        sma_20 = latest.get('sma_20', 0)
        sma_40 = latest.get('sma_40', 0)
        rsi = latest.get('rsi_14', 50)
        
        # テクニカル判断
        price_above_sma20 = current_price > sma_20 if sma_20 > 0 else False
        golden_cross = sma_20 > sma_40 if sma_20 > 0 and sma_40 > 0 else False
        oversold = rsi < 30
        overbought = rsi > 70
    else:
        price_above_sma20 = False
        golden_cross = False
        oversold = False
        overbought = False
    
    # 投資判断ロジック（新閾値: 57点以上=買い、50-56点=推奨）
    if total_score >= 57:  # 強い買い意識が必要な閾値
        if momentum_score >= 15 and price_above_sma20 and golden_cross:
            recommendation['action'] = '強い買い'
            recommendation['reasoning'] = '高スコア（57点以上）+ 強いモメンタム + 良好なテクニカル'
            recommendation['entry_strategy'] = '現在価格での即座エントリー推奨'
        elif momentum_score >= 10:
            recommendation['action'] = '買い'
            recommendation['reasoning'] = '高スコア（57点以上）+ 安定したモメンタム'
            recommendation['entry_strategy'] = '押し目での段階的エントリー'
        else:
            recommendation['action'] = '買い'
            recommendation['reasoning'] = '高スコア（57点以上）達成、強い買い推奨レベル'
            recommendation['entry_strategy'] = '分割エントリーで積極的に取得'
    
    elif total_score >= 50:  # 推奨レベル（50-56点）
        if value_score >= 12 and oversold:
            recommendation['action'] = '推奨'
            recommendation['reasoning'] = '推奨スコア（50-56点）+ 割安 + 売られ過ぎからの反発期待'
            recommendation['entry_strategy'] = '押し目での分割エントリー検討'
        elif momentum_score >= 12:
            recommendation['action'] = '推奨'
            recommendation['reasoning'] = '推奨スコア（50-56点）+ 良好なモメンタム'
            recommendation['entry_strategy'] = '慎重な段階的エントリー'
        elif total_score >= 54:
            recommendation['action'] = '推奨'
            recommendation['reasoning'] = '推奨スコア上位（54点以上）、バランス良好'
            recommendation['entry_strategy'] = '小ポジションから段階的エントリー'
        else:
            recommendation['action'] = '推奨'
            recommendation['reasoning'] = '推奨スコア（50-56点）達成、検討価値あり'
            recommendation['entry_strategy'] = '様子見しながら小ポジション検討'
    
    elif total_score >= 45:  # 従来の基準を維持
        if value_score >= 12 and oversold:
            recommendation['action'] = '条件付き買い'
            recommendation['reasoning'] = 'スコア45点以上 + 割安 + 売られ過ぎからの反発期待'
            recommendation['entry_strategy'] = '慎重な分割エントリー'
        elif momentum_score >= 12:
            recommendation['action'] = '条件付き買い'
            recommendation['reasoning'] = 'スコア45点以上 + 良好なモメンタム'
            recommendation['entry_strategy'] = '小ポジションでの様子見エントリー'
        else:
            recommendation['action'] = '弱い買い'
            recommendation['reasoning'] = 'スコア45点以上だが慎重に検討'
            recommendation['entry_strategy'] = 'より良いエントリーポイントを待つ'
    
    else:
        recommendation['action'] = '見送り'
        recommendation['reasoning'] = 'スコア45点未満、他の選択肢を検討'
        recommendation['entry_strategy'] = 'エントリー非推奨'
    
    # リスクレベル設定（新閾値基準に合わせて調整）
    if total_score >= 57 and momentum_score >= 15:
        recommendation['risk_level'] = '低'
    elif total_score >= 57:
        recommendation['risk_level'] = '低〜中'
    elif total_score >= 50:
        recommendation['risk_level'] = '中'
    elif total_score >= 45:
        recommendation['risk_level'] = '中〜高'
    else:
        recommendation['risk_level'] = '高'
    
    # 投資期間
    if momentum_score >= 15:
        recommendation['time_horizon'] = '短期〜中期（3-12ヶ月）'
    elif value_score >= 12:
        recommendation['time_horizon'] = '中期〜長期（6-24ヶ月）'
    else:
        recommendation['time_horizon'] = '長期（12ヶ月以上）'
    
    # 出口戦略（新閾値基準に合わせて調整）
    if overbought:
        recommendation['exit_strategy'] = 'RSI70超えで部分利確検討'
    elif total_score >= 57:
        recommendation['exit_strategy'] = 'スコア50以下で見直し'
    elif total_score >= 50:
        recommendation['exit_strategy'] = 'スコア45以下または6ヶ月で改善なければ撤退検討'
    elif total_score >= 45:
        recommendation['exit_strategy'] = 'スコア40以下または3ヶ月で改善なければ撤退検討'
    else:
        recommendation['exit_strategy'] = 'スコア改善なければ早期撤退'
    
    return recommendation


def create_five_factor_radar(score_data: pd.Series) -> go.Figure:
    """
    5ファクター分析のレーダーチャートを作成
    
    Args:
        score_data: スコアデータ
        
    Returns:
        Plotlyの図
    """
    # ファクターと実際のスコア
    factors = ['Value', 'Growth', 'Quality', 'Momentum', 'Macro']
    values = [
        score_data.get('value_score', 0),
        score_data.get('growth_score', 0),
        score_data.get('quality_score', 0),
        score_data.get('momentum_score', 0),
        score_data.get('macro_sector_score', 0)
    ]
    
    # 各ファクターの最高点（ユーザー指定通り）
    max_values = [20, 20, 25, 20, 15]  # Value, Growth, Quality, Momentum, Macro
    
    # 満点のベースライン（薄い灰色）
    baseline_trace = go.Scatterpolar(
        r=max_values,
        theta=factors,
        fill='toself',
        name='満点基準',
        line=dict(color='lightgray', width=1),
        fillcolor='rgba(211,211,211,0.2)',
        opacity=0.5
    )
    
    # 実際のスコア（青色）
    score_trace = go.Scatterpolar(
        r=values,
        theta=factors,
        fill='toself',
        name='実際スコア',
        line=dict(color='rgb(55, 126, 184)', width=2),
        fillcolor='rgba(55, 126, 184, 0.3)',
        marker=dict(size=8, color='rgb(55, 126, 184)')
    )
    
    fig = go.Figure(data=[baseline_trace, score_trace])
    
    # レイアウト調整
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max_values) + 2],  # 最大値を各ファクターの最高点に基づいて設定
                gridcolor="lightgray",
                gridwidth=1,
                tickcolor="gray",
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="black"),
                gridcolor="lightgray"
            )
        ),
        showlegend=True,
        title=dict(
            text="5ファクター分析",
            x=0.5,
            font=dict(size=14, color="black")
        ),
        font=dict(color="black"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    
    return fig


def create_enhanced_stock_detail_chart(symbol: str, stock_data: pd.DataFrame, score_data: pd.Series,
                                      weekly_data: pd.DataFrame, financial_metrics: pd.DataFrame,
                                      sector_comparison: pd.DataFrame, technical_data: pd.DataFrame,
                                      basic_info: Dict, score_history: pd.DataFrame = pd.DataFrame(),
                                      fundamental_data: pd.DataFrame = pd.DataFrame(),
                                      engine=None) -> go.Figure:
    """
    株式の詳細分析チャートを作成（5x3グリッド）
    
    Args:
        symbol: 銘柄コード
        stock_data: 株価データ
        score_data: スコアデータ
        weekly_data: 週次データ
        financial_metrics: 財務指標データ
        sector_comparison: セクター比較データ
        technical_data: テクニカルデータ
        basic_info: 基本情報
        score_history: スコア履歴データ
        fundamental_data: 財務データ
        
    Returns:
        Plotlyの図
    """
    # 7x2のサブプロット作成（期間軸でのグラフ分類を考慮）
    subplot_titles = [
        # Row 1: スコア分析（短期）
        '5ファクター分析', '総合スコア推移（1年）',
        # Row 2: 株価分析（短期・中期）
        '日次株価・移動平均（3ヶ月）', '週次株価・26週MA（1年）',
        # Row 3: テクニカル・短期分析
        'テクニカル指標（RSI・MACD）', 'セクター比較（3ヶ月）',
        # Row 4: 出来高・成長性分析
        '出来高分析（3ヶ月）', '収益・利益成長（5年）',
        # Row 5: 財務効率・安全性（中長期）
        '利益率品質（5年）', 'キャッシュ生成・ROE/ROIC（5年）',
        # Row 6: リスク・バリュエーション（中長期）
        '希薄化リスク（株式数・5年）', '財務安全性（5年）',
        # Row 7: 長期財務分析
        'バリュエーション比較（5年）', '財務指標推移（5年）'
    ]
    
    fig = make_subplots(
        rows=7, cols=2,
        subplot_titles=subplot_titles,
        specs=[
            [{"type": "scatterpolar"}, {"type": "xy", "secondary_y": True}],  # スコア推移に2軸
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "xy", "secondary_y": True}, {"type": "scatter"}],  # テクニカル指標を2軸に
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.06,  # 行間隔を少し狭める
        horizontal_spacing=0.08  # 列間隔を狭める
    )
    
    try:
        # 1. 5ファクターレーダーチャート（Row 1, Col 1）
        factors = ['Value', 'Growth', 'Quality', 'Momentum', 'Macro']
        values = [
            score_data.get('value_score', 0),
            score_data.get('growth_score', 0),
            score_data.get('quality_score', 0),
            score_data.get('momentum_score', 0),
            score_data.get('macro_sector_score', 0)
        ]
        max_values = [20, 20, 25, 20, 15]
        
        # 満点のベースライン
        fig.add_trace(go.Scatterpolar(
            r=max_values,
            theta=factors,
            fill='toself',
            name='満点基準',
            line=dict(color='lightgray', width=1),
            fillcolor='rgba(211,211,211,0.2)',
            opacity=0.5
        ), row=1, col=1)
        
        # 実際のスコア
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=factors,
            fill='toself',
            name='実際スコア',
            line=dict(color='rgb(55, 126, 184)', width=2),
            fillcolor='rgba(55, 126, 184, 0.3)',
            marker=dict(size=8, color='rgb(55, 126, 184)')
        ), row=1, col=1)
        
        # レーダーチャートの軸設定
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, max(max_values) + 2],
                gridcolor="lightgray",
                gridwidth=1,
                tickcolor="gray"
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="black"),
                gridcolor="lightgray"
            )
        )
        
    except Exception as e:
        print(f"レーダーチャート作成エラー {symbol}: {e}")
    
    try:
        # 2. 日次株価・移動平均（Row 2, Col 1）
        if not stock_data.empty and 'date' in stock_data.columns and 'close' in stock_data.columns and len(stock_data) > 1:
            # 🔧 修正: データが十分にある場合のみ描画
            # 🔧 デバッグ: プロット前のデータ確認
            print(f"=== {symbol} 株価チャート用データ確認 ===")
            print(f"stock_dataのタイプ: {type(stock_data)}")
            print(f"stock_dataの形状: {stock_data.shape}")
            print(f"stock_dataのカラム: {list(stock_data.columns)}")
            if 'close' in stock_data.columns:
                close_data = stock_data['close'].dropna()
                print(f"Close価格データ統計:")
                print(f"  件数: {len(close_data)}")
                print(f"  範囲: {close_data.min():.2f} - {close_data.max():.2f}")
                print(f"  平均: {close_data.mean():.2f}")
                print(f"  最新5件の値:")
                print(stock_data[['date', 'close']].tail().to_string())
            
            fig.add_trace(
                go.Scatter(x=stock_data['date'], y=stock_data['close'],
                          mode='lines', name='株価', line=dict(color='black', width=2)),
                row=2, col=1
            )
            
            if 'sma_20' in stock_data.columns and stock_data['sma_20'].notna().any():
                sma20_data = stock_data['sma_20'].dropna()
                print(f"SMA20データ: {len(sma20_data)}件, 範囲: {sma20_data.min():.2f} - {sma20_data.max():.2f}")
                fig.add_trace(
                    go.Scatter(x=stock_data['date'], y=stock_data['sma_20'],
                              mode='lines', name='SMA20', line=dict(color='blue', width=1)),
                    row=2, col=1
                )
            
            if 'sma_40' in stock_data.columns and stock_data['sma_40'].notna().any():
                sma40_data = stock_data['sma_40'].dropna()
                print(f"SMA40データ: {len(sma40_data)}件, 範囲: {sma40_data.min():.2f} - {sma40_data.max():.2f}")
                fig.add_trace(
                    go.Scatter(x=stock_data['date'], y=stock_data['sma_40'],
                              mode='lines', name='SMA40', line=dict(color='red', width=1)),
                    row=2, col=1
                )
            print("=" * 50)
        else:
            # 🔧 修正: データ不足の場合は基本情報のみ表示
            print(f"⚠️ {symbol}: stock_dataが不十分です")
            print(f"  Empty: {stock_data.empty}")
            if not stock_data.empty:
                print(f"  カラム: {list(stock_data.columns)}")
                print(f"  形状: {stock_data.shape}")
            
            if basic_info.get('current_price', 0) > 0:
                from datetime import datetime
                today = dt.now()
                print(f"  現在価格をプロット: {basic_info['current_price']}")
                fig.add_trace(
                    go.Scatter(x=[today], y=[basic_info['current_price']],
                              mode='markers', name='現在価格', 
                              marker=dict(size=10, color='red')),
                    row=2, col=1
                )
    except Exception as e:
        print(f"日次株価チャート作成エラー {symbol}: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # 3. 週次株価・26週MA（Row 2, Col 2）
        if not weekly_data.empty and 'date' in weekly_data.columns and 'close' in weekly_data.columns and len(weekly_data) > 1:
            # 🔧 修正: データが十分にある場合のみ描画
            fig.add_trace(
                go.Scatter(x=weekly_data['date'], y=weekly_data['close'],
                          mode='lines', name='週次株価', line=dict(color='darkblue', width=2)),
                row=2, col=2
            )
            
            if 'sma_26w' in weekly_data.columns and weekly_data['sma_26w'].notna().any():
                fig.add_trace(
                    go.Scatter(x=weekly_data['date'], y=weekly_data['sma_26w'],
                              mode='lines', name='26週MA', line=dict(color='orange', width=1)),
                    row=2, col=2
                )
        else:
            # 🔧 修正: 週次データ不足の場合のフォールバック
            print(f"週次データが不足しています {symbol}: データ件数={len(weekly_data) if not weekly_data.empty else 0}")
    except Exception as e:
        print(f"週次株価チャート作成エラー {symbol}: {e}")
    
    try:
        # 4-6. 財務分析チャート（Row 4-7に分散配置）
        if not financial_metrics.empty and 'date' in financial_metrics.columns and len(financial_metrics) > 1:
            # 🔧 修正: データが十分にある場合のみ描画
            # 収益・利益成長（Row 4, Col 2）- 3年表示
            if 'revenue' in financial_metrics.columns and financial_metrics['revenue'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['revenue'],
                              mode='lines+markers', name='売上高', line=dict(color='green')),
                    row=4, col=2
                )
            
            if 'net_income' in financial_metrics.columns and financial_metrics['net_income'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['net_income'],
                              mode='lines+markers', name='純利益', line=dict(color='blue')),
                    row=4, col=2
                )
            
            # 利益率品質（Row 5, Col 1）- 3年表示
            if 'operating_margin' in financial_metrics.columns and financial_metrics['operating_margin'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['operating_margin'],
                              mode='lines+markers', name='営業利益率', line=dict(color='purple')),
                    row=5, col=1
                )
            
            if 'net_margin' in financial_metrics.columns and financial_metrics['net_margin'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['net_margin'],
                              mode='lines+markers', name='純利益率', line=dict(color='orange')),
                    row=5, col=1
                )
            
            # キャッシュ生成・ROE/ROIC（Row 5, Col 2）- 5年表示
            if 'fcf_yield' in financial_metrics.columns and financial_metrics['fcf_yield'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['fcf_yield'],
                              mode='lines+markers', name='FCF利回り', line=dict(color='darkgreen')),
                    row=5, col=2
                )
            
            if 'roe' in financial_metrics.columns and financial_metrics['roe'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['roe'],
                              mode='lines+markers', name='ROE', line=dict(color='red')),
                    row=5, col=2
                )
            
            if 'roic' in financial_metrics.columns and financial_metrics['roic'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['roic'],
                              mode='lines+markers', name='ROIC', line=dict(color='blue')),
                    row=5, col=2
                )
            
            # 希薄化リスク（Row 6, Col 1） - 5年表示
            if 'float_shares' in financial_metrics.columns and financial_metrics['float_shares'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['float_shares'],
                               mode='lines+markers', name='浮動株数', line=dict(color='lightcoral')),
                    row=6, col=1
                )
            
            if 'outstanding_shares' in financial_metrics.columns and financial_metrics['outstanding_shares'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['outstanding_shares'],
                               mode='lines+markers', name='発行済株式数', line=dict(color='darkred')),
                    row=6, col=1
                )
            
            # 財務安全性（Row 6, Col 2） - 5年表示
            print(f"=== {symbol} 財務安全性データ確認 ===")
            print(f"financial_metrics カラム: {list(financial_metrics.columns) if not financial_metrics.empty else '空のデータフレーム'}")
            print(f"debt_to_equity カラムの存在: {'✅' if not financial_metrics.empty and 'debt_to_equity' in financial_metrics.columns else '❌'}")
            print(f"current_ratio カラムの存在: {'✅' if not financial_metrics.empty and 'current_ratio' in financial_metrics.columns else '❌'}")
            
            if not financial_metrics.empty and 'debt_to_equity' in financial_metrics.columns:
                de_data = financial_metrics['debt_to_equity'].dropna()
                print(f"D/E比率データ: {len(de_data)}件 (値の範囲: {de_data.min():.2f} - {de_data.max():.2f})" if len(de_data) > 0 else "D/E比率データ: 有効データなし")
                if financial_metrics['debt_to_equity'].notna().any():
                    fig.add_trace(
                        go.Scatter(x=financial_metrics['date'], y=financial_metrics['debt_to_equity'],
                                  mode='lines+markers', name='D/E比率', line=dict(color='darkred')),
                        row=6, col=2
                    )
                    print(f"✅ D/E比率グラフを追加しました")
            else:
                print(f"❌ D/E比率データが見つかりません")
            
            if not financial_metrics.empty and 'current_ratio' in financial_metrics.columns:
                cr_data = financial_metrics['current_ratio'].dropna()
                print(f"流動比率データ: {len(cr_data)}件 (値の範囲: {cr_data.min():.2f} - {cr_data.max():.2f})" if len(cr_data) > 0 else "流動比率データ: 有効データなし")
                if financial_metrics['current_ratio'].notna().any():
                    fig.add_trace(
                        go.Scatter(x=financial_metrics['date'], y=financial_metrics['current_ratio'],
                                  mode='lines+markers', name='流動比率', line=dict(color='green')),
                        row=6, col=2
                    )
                    print(f"✅ 流動比率グラフを追加しました")
            else:
                print(f"❌ 流動比率データが見つかりません")
            
            # バリュエーション比較（Row 7, Col 1） - 5年表示
            if 'ev_ebitda' in financial_metrics.columns and financial_metrics['ev_ebitda'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['ev_ebitda'],
                              mode='lines+markers', name='EV/EBITDA', line=dict(color='navy')),
                    row=7, col=1
                )
            
            # PERの追加（fundamental_dataから）
            if not fundamental_data.empty and 'per' in fundamental_data.columns and fundamental_data['per'].notna().any():
                fig.add_trace(
                    go.Scatter(x=fundamental_data['date'], y=fundamental_data['per'],
                              mode='lines+markers', name='PER', line=dict(color='purple')),
                    row=7, col=1
                )
            
            # 財務指標推移（Row 7, Col 2） - 5年表示
            if 'debt_to_equity' in financial_metrics.columns and financial_metrics['debt_to_equity'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['debt_to_equity'],
                              mode='lines+markers', name='D/E', line=dict(color='red')),
                    row=7, col=2
                )
                
            if 'return_on_assets' in financial_metrics.columns and financial_metrics['return_on_assets'].notna().any():
                fig.add_trace(
                    go.Scatter(x=financial_metrics['date'], y=financial_metrics['return_on_assets'],
                              mode='lines+markers', name='ROA', line=dict(color='green')),
                    row=7, col=2
                )
            
            # fundamental_dataからも財務指標を追加
            if not fundamental_data.empty:
                if 'roic' in fundamental_data.columns and fundamental_data['roic'].notna().any():
                    fig.add_trace(
                        go.Scatter(x=fundamental_data['date'], y=fundamental_data['roic'] * 100,
                                  mode='lines+markers', name='ROIC(%)', line=dict(color='blue')),
                        row=7, col=2
                    )
                
                if 'debt_to_equity' in fundamental_data.columns and fundamental_data['debt_to_equity'].notna().any():
                    fig.add_trace(
                        go.Scatter(x=fundamental_data['date'], y=fundamental_data['debt_to_equity'],
                                  mode='lines+markers', name='D/E', line=dict(color='red')),
                        row=7, col=2
                    )
        else:
            # 🔧 修正: 財務データ不足の場合のフォールバック
            print(f"財務データが不足しています {symbol}: データ件数={len(financial_metrics) if not financial_metrics.empty else 0}")
    except Exception as e:
        print(f"財務分析チャート作成エラー {symbol}: {e}")
    
    try:
        # 7-9. セクター比較・テクニカル分析・出来高分析（Row 3-4に移動）
        # 🔧 デバッグ: データ状況を詳細確認
        print(f"=== {symbol} チャート作成データ確認 ===")
        print(f"technical_data: {'✅' if not technical_data.empty else '❌'} ({len(technical_data)}件)")
        print(f"sector_comparison: {'✅' if not sector_comparison.empty else '❌'} ({len(sector_comparison)}件)")
        print(f"financial_metrics: {'✅' if not financial_metrics.empty else '❌'} ({len(financial_metrics)}件)")
        print(f"weekly_data: {'✅' if not weekly_data.empty else '❌'} ({len(weekly_data)}件)")
        print(f"score_history: {'✅' if not score_history.empty else '❌'} ({len(score_history)}件)")
        
        if not technical_data.empty:
            print(f"テクニカルデータのカラム: {list(technical_data.columns)}")
            print(f"必要カラム確認 - rsi_14: {'✅' if 'rsi_14' in technical_data.columns else '❌'}")
            print(f"必要カラム確認 - macd_hist: {'✅' if 'macd_hist' in technical_data.columns else '❌'}")
            print(f"必要カラム確認 - volume: {'✅' if 'volume' in technical_data.columns else '❌'}")
        
        # セクター比較（株価）（Row 3, Col 2）- 位置変更と改良
        if not sector_comparison.empty:
            # セクターデータの最新日を確認
            sector_latest = sector_comparison['date'].max()
            from datetime import datetime as dt
            current_date = dt.now().date()
            
            # 日付型の処理
            if hasattr(sector_latest, 'date'):
                sector_latest_date = sector_latest.date()
            else:
                sector_latest_date = sector_latest
                
            days_behind = (current_date - sector_latest_date).days
            
            # セクターデータが5日以上古い場合は警告
            if days_behind > 5:
                print(f"Warning: セクターデータが{days_behind}日古いです（最新: {sector_latest_date}）")
            
            # 個別株価データを直接取得（セクターデータの期間に制限）
            individual_stock_query = text("""
            SELECT 
                date,
                close
            FROM fmp_data.daily_prices
            WHERE symbol = :symbol
              AND date >= :start_date
              AND date <= :sector_end_date
            ORDER BY date
            """)
            
            try:
                # セクターデータの期間に制限して個別株価データを取得
                start_date = (dt.now() - timedelta(days=126)).strftime('%Y-%m-%d')
                sector_end_date = sector_latest_date.strftime('%Y-%m-%d')
                
                individual_prices = pd.read_sql(individual_stock_query, engine, 
                                              params={"symbol": symbol, "start_date": start_date, "sector_end_date": sector_end_date})
                
                if not individual_prices.empty:
                    print(f"Debug: {symbol} 個別株価データ - {len(individual_prices)}行, 価格範囲: {individual_prices['close'].min():.2f} - {individual_prices['close'].max():.2f}")
                    print(f"Debug: 最新個別株価 - 日付: {individual_prices['date'].max()}, 価格: {individual_prices['close'].iloc[-1]:.2f}")
                    print(f"Debug: 最古個別株価 - 日付: {individual_prices['date'].min()}, 価格: {individual_prices['close'].iloc[0]:.2f}")
                    
                    # データの期間を合わせる（共通の日付のみ使用）
                    stock_dates = individual_prices['date']
                    sector_dates = sector_comparison['date']
                    
                    print(f"Debug: セクターデータ期間 - {sector_dates.min()} ～ {sector_dates.max()}")
                    print(f"Debug: 個別株価期間 - {stock_dates.min()} ～ {stock_dates.max()}")
                    
                    # 共通の日付範囲を取得
                    min_date = max(stock_dates.min(), sector_dates.min())
                    max_date = min(stock_dates.max(), sector_dates.max())
                    
                    print(f"Debug: 共通期間 - {min_date} ～ {max_date}")
                    
                    # 期間でフィルタリング
                    stock_filtered = individual_prices[
                        (individual_prices['date'] >= min_date) & 
                        (individual_prices['date'] <= max_date)
                    ].copy()
                    sector_filtered = sector_comparison[
                        (sector_comparison['date'] >= min_date) & 
                        (sector_comparison['date'] <= max_date)
                    ].copy()
                    
                    print(f"Debug: フィルタ後 - 個別株価: {len(stock_filtered)}行, セクター: {len(sector_filtered)}行")
                    if not stock_filtered.empty:
                        print(f"Debug: フィルタ後個別株価 - 最新: {stock_filtered['date'].max()}, 価格: {stock_filtered['close'].iloc[-1]:.2f}")
                    if not sector_filtered.empty:
                        print(f"Debug: フィルタ後セクター - 最新: {sector_filtered['date'].max()}, 価格: {sector_filtered['avg_close'].iloc[-1]:.2f}")
                    
                    if not stock_filtered.empty and not sector_filtered.empty and len(stock_filtered) > 0 and len(sector_filtered) > 0:
                        # 個別株価（正規化）
                        stock_prices = stock_filtered['close'].dropna()
                        
                        if len(stock_prices) > 0:
                            # 最初の有効な株価で正規化
                            first_valid_price = stock_prices.iloc[0]
                            stock_normalized = (stock_prices / first_valid_price) * 100
                            
                            fig.add_trace(
                                go.Scatter(x=stock_filtered['date'][:len(stock_normalized)], y=stock_normalized,
                                          mode='lines', name=f'{symbol}', line=dict(color='blue', width=2)),
                                row=3, col=2
                            )
                        
                        # セクター平均（正規化）
                        sector_prices = sector_filtered['avg_close'].dropna()
                        
                        if len(sector_prices) > 0:
                            # 最初の有効なセクター価格で正規化
                            first_valid_sector = sector_prices.iloc[0]
                            if first_valid_sector > 0:  # ゼロ除算を避ける
                                sector_normalized = (sector_prices / first_valid_sector) * 100
                                
                                fig.add_trace(
                                    go.Scatter(x=sector_filtered['date'][:len(sector_normalized)], y=sector_normalized,
                                              mode='lines', name='セクター平均', line=dict(color='gray', width=1)),
                                    row=3, col=2
                                )
                                
                # チャートタイトルを更新してデータ期間を明示
                if days_behind > 5:
                    fig.layout.annotations[5].text = f"セクター比較（3ヶ月）※データ制限あり"
                else:
                    fig.layout.annotations[5].text = f"セクター比較（3ヶ月）"
                        
            except Exception as e:
                print(f"セクター比較用個別株価取得エラー {symbol}: {e}")
                # エラーの場合もタイトルを更新
                fig.layout.annotations[5].text = "セクター比較（データエラー）"
        
        # セクターデータが利用できない場合の処理を追加
        if sector_comparison.empty:
            print(f"Warning: {symbol} のセクター比較データが利用できません")
            fig.layout.annotations[5].text = "セクター比較（データなし）"
        
        # テクニカル指標（Row 3, Col 1）- 位置変更
        if not technical_data.empty:
            print(f"📊 テクニカル指標チャート作成中...")
            # RSI（メイン軸、50を中心に20-80の範囲で表示）
            if 'rsi_14' in technical_data.columns and technical_data['rsi_14'].notna().any():
                rsi_data = technical_data['rsi_14'].dropna()
                print(f"RSI描画: {len(rsi_data)}件, 範囲: {rsi_data.min():.1f} - {rsi_data.max():.1f}")
                fig.add_trace(
                    go.Scatter(x=technical_data['date'], y=technical_data['rsi_14'],
                              mode='lines', name='RSI', line=dict(color='purple')),
                    row=3, col=1, secondary_y=False
                )
                
                # RSI基準線（shape使用）
                fig.add_shape(
                    type="line",
                    x0=technical_data['date'].min() if not technical_data.empty else 0,
                    y0=70,
                    x1=technical_data['date'].max() if not technical_data.empty else 1,
                    y1=70,
                    line=dict(color="red", width=1, dash="dash"),
                    row=3, col=1
                )
                fig.add_shape(
                    type="line",
                    x0=technical_data['date'].min() if not technical_data.empty else 0,
                    y0=30,
                    x1=technical_data['date'].max() if not technical_data.empty else 1,
                    y1=30,
                    line=dict(color="green", width=1, dash="dash"),
                    row=3, col=1
                )
                fig.add_shape(
                    type="line",
                    x0=technical_data['date'].min() if not technical_data.empty else 0,
                    y0=50,
                    x1=technical_data['date'].max() if not technical_data.empty else 1,
                    y1=50,
                    line=dict(color="gray", width=1, dash="dot"),
                    row=3, col=1
                )
            
            # MACD histogram（セカンダリ軸、0中心）
            if 'macd_hist' in technical_data.columns and technical_data['macd_hist'].notna().any():
                macd_data = technical_data['macd_hist'].dropna()
                print(f"MACD描画: {len(macd_data)}件, 範囲: {macd_data.min():.3f} - {macd_data.max():.3f}")
                fig.add_trace(
                    go.Bar(x=technical_data['date'], y=technical_data['macd_hist'],
                           name='MACD Hist', marker_color='orange', opacity=0.7),
                    row=3, col=1, secondary_y=True
                )
                
                # MACD 0基準線
                fig.add_shape(
                    type="line",
                    x0=technical_data['date'].min() if not technical_data.empty else 0,
                    y0=0,
                    x1=technical_data['date'].max() if not technical_data.empty else 1,
                    y1=0,
                    line=dict(color="black", width=1, dash="dot"),
                    row=3, col=1
                )
            else:
                print(f"⚠️ MACDデータなし: カラム存在={('macd_hist' in technical_data.columns)}")
        else:
            print(f"⚠️ technical_dataが空です")
        
        # Y軸の設定
        # RSI軸（メイン軸）: 20-80の範囲、50を中心
        fig.update_yaxes(title_text="RSI", range=[20, 80], row=3, col=1, secondary_y=False)
        
        # MACD軸（セカンダリ軸）: 0を中心とした適切な範囲
        if not technical_data.empty and 'macd_hist' in technical_data.columns:
            macd_max = technical_data['macd_hist'].max()
            macd_min = technical_data['macd_hist'].min()
            macd_range = max(abs(macd_max), abs(macd_min))
            fig.update_yaxes(title_text="MACD Hist", range=[-macd_range*1.1, macd_range*1.1], 
                           row=3, col=1, secondary_y=True)
        
        # 出来高分析（Row 4, Col 1）- 位置変更
        print(f"📊 出来高分析チャート作成中...")
        if not technical_data.empty and 'volume' in technical_data.columns:
            volume_data = technical_data['volume'].dropna()
            if len(volume_data) > 0:
                print(f"出来高描画: {len(volume_data)}件, 範囲: {volume_data.min():,.0f} - {volume_data.max():,.0f}")
                fig.add_trace(
                    go.Bar(x=technical_data['date'], y=technical_data['volume'],
                           name='出来高', marker_color='lightblue'),
                    row=4, col=1
                )
            else:
                print(f"⚠️ 出来高データが空です")
        else:
            print(f"⚠️ 出来高データなし - technical_data空: {technical_data.empty}, volumeカラム: {'volume' in technical_data.columns if not technical_data.empty else False}")
        
        # 財務指標推移（Row 7, Col 2）- 位置変更
        if not fundamental_data.empty:
            if 'roe' in fundamental_data.columns:
                fig.add_trace(
                    go.Scatter(x=fundamental_data['date'], y=fundamental_data['roe'] * 100,
                              mode='lines+markers', name='ROE(%)', line=dict(color='green')),
                    row=7, col=2
                )
            
            if 'roic' in fundamental_data.columns:
                fig.add_trace(
                    go.Scatter(x=fundamental_data['date'], y=fundamental_data['roic'] * 100,
                              mode='lines+markers', name='ROIC(%)', line=dict(color='blue')),
                    row=7, col=2
                )
            
            if 'debt_to_equity' in fundamental_data.columns:
                fig.add_trace(
                    go.Scatter(x=fundamental_data['date'], y=fundamental_data['debt_to_equity'],
                              mode='lines+markers', name='D/E', line=dict(color='red')),
                    row=7, col=2
                )
        
        # 総合スコア推移（Row 1, Col 2） - 5ファクターと隣合わせ、改良版
        # score_historyがある場合は実際のスコア履歴を表示
        if not score_history.empty:
            # 各ファクタースコア（プライマリ軸、見やすい色、適度な太さ）
            if 'value_score' in score_history.columns:
                fig.add_trace(
                    go.Scatter(x=score_history['date'], y=score_history['value_score'],
                              mode='lines', name='Value', line=dict(color='darkred', width=2),
                              opacity=0.9),
                    row=1, col=2, secondary_y=False
                )
                
            if 'growth_score' in score_history.columns:
                fig.add_trace(
                    go.Scatter(x=score_history['date'], y=score_history['growth_score'],
                              mode='lines', name='Growth', line=dict(color='darkgreen', width=2),
                              opacity=0.9),
                    row=1, col=2, secondary_y=False
                )
                
            if 'quality_score' in score_history.columns:
                fig.add_trace(
                    go.Scatter(x=score_history['date'], y=score_history['quality_score'],
                              mode='lines', name='Quality', line=dict(color='darkblue', width=2),
                              opacity=0.9),
                    row=1, col=2, secondary_y=False
                )
                
            if 'momentum_score' in score_history.columns:
                fig.add_trace(
                    go.Scatter(x=score_history['date'], y=score_history['momentum_score'],
                              mode='lines', name='Momentum', line=dict(color='orange', width=1.5),
                              opacity=0.8),
                    row=1, col=2, secondary_y=False
                )
                
            if 'macro_sector_score' in score_history.columns:
                fig.add_trace(
                    go.Scatter(x=score_history['date'], y=score_history['macro_sector_score'],
                              mode='lines', name='Macro', line=dict(color='purple', width=2),
                              opacity=0.9),
                    row=1, col=2, secondary_y=False
                )
            
            # 総合スコア（セカンダリ軸、太い線、目立つ色）
            if 'total_score' in score_history.columns:
                fig.add_trace(
                    go.Scatter(x=score_history['date'], y=score_history['total_score'],
                              mode='lines+markers', name='総合スコア', 
                              line=dict(color='darkblue', width=3),
                              marker=dict(size=4, color='darkblue')),
                    row=1, col=2, secondary_y=True
                )
        else:
            # フォールバック: fundamental_dataを使ってスコアの代替指標を表示
            if not fundamental_data.empty:
                # FCF Yield
                if 'fcf_yield' in fundamental_data.columns:
                    fig.add_trace(
                        go.Scatter(x=fundamental_data['date'], y=fundamental_data['fcf_yield'],
                                  mode='lines+markers', name='FCF Yield', line=dict(color='darkgreen')),
                        row=1, col=2, secondary_y=False
                    )
                
                # EPS CAGR
                if 'eps_cagr_3y' in fundamental_data.columns:
                    fig.add_trace(
                        go.Scatter(x=fundamental_data['date'], y=fundamental_data['eps_cagr_3y'],
                                  mode='lines+markers', name='EPS CAGR 3Y', line=dict(color='purple')),
                        row=1, col=2, secondary_y=False
                    )
        
        # Y軸の設定
        # ファクタースコア軸（プライマリ軸）: 0-25の範囲
        fig.update_yaxes(title_text="ファクタースコア", range=[0, 25], row=1, col=2, secondary_y=False)
        
        # 総合スコア軸（セカンダリ軸）: 0-100の範囲
        fig.update_yaxes(title_text="総合スコア", range=[0, 100], row=1, col=2, secondary_y=True)
    except Exception as e:
        print(f"テクニカル・出来高チャート作成エラー {symbol}: {e}")
    
    # レイアウト調整
    fig.update_layout(
        height=2200,  # 高さを大幅に増加（1800→2200）
        title=f"{symbol} - 包括的分析ダッシュボード",
        showlegend=False,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    return fig


def create_basic_fallback_chart(symbol: str, score_data: pd.Series, fundamental_data: pd.DataFrame,
                               technical_data: pd.DataFrame, basic_info: Dict) -> go.Figure:
    """
    データが不足している場合のフォールバック用チャート（改善版）
    
    Args:
        symbol: 銘柄コード
        score_data: スコアデータ
        fundamental_data: 財務データ
        technical_data: テクニカルデータ
        basic_info: 基本情報
        
    Returns:
        Plotlyの図
    """
    # 🔧 修正: 2x3のサブプロット作成（より多くの情報を表示）
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            '5ファクター分析', '株価推移（価格＋移動平均）', 'RSI指標（過熱感）',
            '出来高推移', '基本情報', 'スコア詳細'
        ],
        specs=[
            [{"type": "scatterpolar"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "table"}, {"type": "bar"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    # 1. 5ファクターレーダーチャート
    factors = ['Value', 'Growth', 'Quality', 'Momentum', 'Macro']
    values = [
        score_data.get('value_score', 0),
        score_data.get('growth_score', 0),
        score_data.get('quality_score', 0),
        score_data.get('momentum_score', 0),
        score_data.get('macro_sector_score', 0)
    ]
    max_values = [20, 20, 25, 20, 15]
    
    # 満点のベースライン
    fig.add_trace(go.Scatterpolar(
        r=max_values,
        theta=factors,
        fill='toself',
        name='満点',
        line_color='lightgray',
        fillcolor='rgba(211,211,211,0.2)'
    ), row=1, col=1)
    
    # 実際のスコア
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=factors,
        fill='toself',
        name='実際スコア',
        line_color='rgb(55, 126, 184)',
        fillcolor='rgba(55, 126, 184, 0.3)'
    ), row=1, col=1)
    
    # 2. 株価推移（改善版）
    if not technical_data.empty and 'close' in technical_data.columns and len(technical_data) > 1:
        # 🔧 修正: データが十分にある場合のみ描画
        fig.add_trace(
            go.Scatter(x=technical_data['date'], y=technical_data['close'],
                      mode='lines', name='株価', line=dict(color='black', width=2)),
            row=1, col=2
        )
        
        if 'sma_20' in technical_data.columns:
            fig.add_trace(
                go.Scatter(x=technical_data['date'], y=technical_data['sma_20'],
                          mode='lines', name='SMA20', line=dict(color='blue', width=1)),
                row=1, col=2
            )
        
        if 'sma_40' in technical_data.columns:
            fig.add_trace(
                go.Scatter(x=technical_data['date'], y=technical_data['sma_40'],
                          mode='lines', name='SMA40', line=dict(color='red', width=1)),
                row=1, col=2
            )
    else:
        # データ不足の場合は現在価格のみ表示
        current_price = basic_info.get('current_price', 0)
        if current_price > 0:
            from datetime import datetime
            today = datetime.now()
            fig.add_trace(
                go.Scatter(x=[today], y=[current_price],
                          mode='markers', name='現在価格', 
                          marker=dict(size=10, color='red')),
                row=1, col=2
            )
    
    # 3. RSI指標（改善版）
    if not technical_data.empty and 'rsi_14' in technical_data.columns and len(technical_data) > 1:
        fig.add_trace(
            go.Scatter(x=technical_data['date'], y=technical_data['rsi_14'],
                      mode='lines', name='RSI', line=dict(color='purple')),
            row=1, col=3
        )
        
        # RSI基準線を追加
        x_range = [technical_data['date'].min(), technical_data['date'].max()]
        for level, color, dash in [(70, 'red', 'dash'), (30, 'green', 'dash'), (50, 'gray', 'dot')]:
            fig.add_trace(
                go.Scatter(x=x_range, y=[level, level], mode='lines',
                          line=dict(color=color, width=1, dash=dash),
                          name=f'RSI {level}', showlegend=False),
                row=1, col=3
            )
    
    # 4. 出来高推移
    if not technical_data.empty and 'volume' in technical_data.columns and len(technical_data) > 1:
        fig.add_trace(
            go.Scatter(x=technical_data['date'], y=technical_data['volume'],
                      mode='lines', name='出来高', line=dict(color='orange')),
            row=2, col=1
        )
    
    # 5. 基本情報テーブル
    table_data = [
        ['現在価格', f"{basic_info.get('current_price', 'N/A'):.2f}" if basic_info.get('current_price', 0) > 0 else 'N/A'],
        ['時価総額', f"{basic_info.get('market_cap', 0) / 1_000_000:.0f}M" if basic_info.get('market_cap', 0) > 0 else 'N/A'],
        ['PER', f"{basic_info.get('per', 0):.1f}" if basic_info.get('per', 0) > 0 else 'N/A'],
        ['PBR', f"{basic_info.get('pbr', 0):.1f}" if basic_info.get('pbr', 0) > 0 else 'N/A'],
        ['セクター', basic_info.get('sector', 'N/A')],
        ['業界', basic_info.get('industry', 'N/A')]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['項目', '値'], fill_color='lightblue'),
            cells=dict(values=list(zip(*table_data)), fill_color='white')
        ),
        row=2, col=2
    )
    
    # 6. スコア詳細（棒グラフ）
    score_names = ['Value', 'Growth', 'Quality', 'Momentum', 'Macro']
    score_values = values
    
    fig.add_trace(
        go.Bar(x=score_names, y=score_values, name='スコア',
               marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']),
        row=2, col=3
    )
    
    # レイアウト調整
    fig.update_layout(
        height=1000,  # 高さを増加
        title=f"{symbol} - フォールバック分析チャート（データ制限版）",
        showlegend=False
    )
    
    return fig


def generate_top_stocks_report(engine, target_date: str = None) -> str:
    """
    スコア上位銘柄レポートのHTMLを生成（日本株・米国株分離版）
    
    Args:
        engine: データベースエンジン
        target_date: 対象日
        
    Returns:
        HTMLコンテンツ
    """
    # 米国株と日本株のスコア上位銘柄を取得（全種類・各5銘柄）
    print("スコア上位銘柄データを取得中...")
    
    # 1. フィルター無し（全銘柄）
    us_top_stocks_all = get_top_scored_stocks_by_market(engine, target_date, top_n=5, market_type='US')
    jp_top_stocks_all = get_top_scored_stocks_by_market(engine, target_date, top_n=5, market_type='JP')
    
    # 2. フィルター適用（良質銘柄のみ）
    us_top_stocks_filtered = get_top_scored_stocks_by_market_filtered(engine, target_date, top_n=5, market_type='US')
    jp_top_stocks_filtered = get_top_scored_stocks_by_market_filtered(engine, target_date, top_n=5, market_type='JP')
    
    # 全てのデータを結合
    top_stocks = pd.concat([
        us_top_stocks_all, jp_top_stocks_all,
        us_top_stocks_filtered, jp_top_stocks_filtered
    ], ignore_index=True)
    
    if top_stocks.empty:
        return "<html><body><h1>データが見つかりません</h1></body></html>"
    
    # 最新日付を取得
    report_date = None
    if not us_top_stocks_all.empty:
        report_date = us_top_stocks_all.iloc[0]['date']
    elif not jp_top_stocks_all.empty:
        report_date = jp_top_stocks_all.iloc[0]['date']
    elif not us_top_stocks_filtered.empty:
        report_date = us_top_stocks_filtered.iloc[0]['date']
    elif not jp_top_stocks_filtered.empty:
        report_date = jp_top_stocks_filtered.iloc[0]['date']
    
    print(f"スコア上位銘柄取得完了:")
    print(f"  米国株(全): {len(us_top_stocks_all)}件")
    print(f"  米国株(フィルター): {len(us_top_stocks_filtered)}件")
    print(f"  日本株(全): {len(jp_top_stocks_all)}件")
    print(f"  日本株(フィルター): {len(jp_top_stocks_filtered)}件")
    
    # HTMLの開始部分
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>スコア上位銘柄分析レポート</title>
        <script src="https://cdn.plot.ly/plotly-2.29.0.min.js"></script>
        {add_simple_watchlist_css()}
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 20px;
            }}
            .stock-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 20px 0;
                padding: 20px;
                background-color: #fff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .stock-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            .stock-title {{
                font-size: 1.5em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .total-score {{
                font-size: 2em;
                font-weight: bold;
                color: #e74c3c;
            }}
            .score-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }}
            .score-item {{
                text-align: center;
                padding: 10px;
                border-radius: 6px;
                background-color: #f8f9fa;
            }}
            .score-label {{
                font-size: 0.9em;
                color: #666;
                margin-bottom: 5px;
            }}
            .score-value {{
                font-size: 1.2em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .analysis-section {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 6px;
            }}
            .strengths {{
                color: #27ae60;
                font-weight: bold;
            }}
            .weaknesses {{
                color: #e74c3c;
                font-weight: bold;
            }}
            .recommendation {{
                background-color: #e8f5e8;
                border-left: 4px solid #27ae60;
                padding: 15px;
                margin: 15px 0;
            }}
            .recommendation.recommend {{
                background-color: #e3f2fd;
                border-left-color: #2196f3;
            }}
            .recommendation.hold {{
                background-color: #fff3cd;
                border-left-color: #ffc107;
            }}
            .recommendation.sell {{
                background-color: #f8d7da;
                border-left-color: #dc3545;
            }}
            .basic-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin: 15px 0;
                padding: 15px;
                background-color: #e3f2fd;
                border-radius: 6px;
            }}
            .info-item {{
                display: flex;
                justify-content: space-between;
            }}
            .chart-container {{
                margin: 20px 0;
                height: 2300px;  # チャート高さ2200px + 余白
                overflow: visible;
            }}
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .summary-table th,
            .summary-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            .summary-table th {{
                background-color: #3498db;
                color: white;
            }}
            .rank-1 {{ background-color: #ffd700; }}
            .rank-2 {{ background-color: #c0c0c0; }}
            .rank-3 {{ background-color: #cd7f32; }}
            
            /* チャートボタンのスタイル */
            .chart-button {{
                background-color: #3498db;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.8em;
                margin: 2px;
                transition: background-color 0.3s;
            }}
            .chart-button:hover {{
                background-color: #2980b9;
            }}
            
            /* 詳細レポートボタンのスタイル */
            .detailed-report-btn {{
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.9em;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: all 0.3s;
            }}
            .detailed-report-btn:hover {{
                background: linear-gradient(135deg, #2980b9 0%, #1e5f8e 100%);
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }}
            .detailed-report-btn:disabled {{
                background: #bdc3c7;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }}
            
            /* ランキングに戻るボタンのスタイル */
            .back-to-ranking-btn {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.9em;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: all 0.3s;
            }}
            .back-to-ranking-btn:hover {{
                background: linear-gradient(135deg, #c0392b 0%, #a93226 100%);
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 スコア上位銘柄分析レポート</h1>
                <p>分析日: {report_date}</p>
                <div style="margin-top: 15px; padding: 10px; background-color: #e8f4f8; border-radius: 6px;">
                    <h3 style="margin: 0 0 10px 0; color: #2c3e50;">📋 ランキング構成</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 0.9em;">
                        <div>
                            <strong>🇺🇸 米国株</strong><br>
                            • 全銘柄ランキング: {len(us_top_stocks_all)}銘柄<br>
                            • 良質銘柄ランキング: {len(us_top_stocks_filtered)}銘柄
                        </div>
                        <div>
                            <strong>🇯🇵 日本株</strong><br>
                            • 全銘柄ランキング: {len(jp_top_stocks_all)}銘柄<br>
                            • 良質銘柄ランキング: {len(jp_top_stocks_filtered)}銘柄
                        </div>
                    </div>
                    <p style="margin: 10px 0 0 0; font-size: 0.8em; color: #666;">
                        💡 良質銘柄ランキング = バリュートラップ・品質成長フィルター適用済み
                    </p>
                </div>
            </div>
    """
    
    # 推奨アクションに基づく色分けを取得する関数
    def get_recommendation_color(action):
        """推奨アクションに基づいて背景色を返す（淡い色合い）"""
        color_map = {
            '強い買い': '#a8e6cf',      # 淡い緑
            '買い': '#d4f6d4',          # より淡い緑
            '推奨': '#cce7ff',          # 淡い青
            '条件付き買い': '#ffe4b3',   # 淡いオレンジ
            '弱い買い': '#fff3b3',      # 淡い黄色
            '見送り': '#e6e6e6'         # 淡いグレー
        }
        return color_map.get(action, '#f8f9fa')  # デフォルトは非常に薄いグレー

    # ランキングテーブル作成関数
    def create_score_ranking_table(data, title, market_icon, subtitle=""):
        subtitle_html = f"<p style='margin: 5px 0 15px 0; color: #666; font-size: 0.9em;'>{subtitle}</p>" if subtitle else ""
        table_html = f"""
            <!-- {title}ランキングテーブル -->
            <h2>🏆 {market_icon} {title}</h2>
            {subtitle_html}
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Watch</th>
                        <th>順位</th>
                        <th>銘柄</th>
                        <th>総合スコア</th>
                        <th>Value</th>
                        <th>Growth</th>
                        <th>Quality</th>
                        <th>Momentum</th>
                        <th>Macro</th>
                        <th>推奨アクション</th>
                        <th>チャート</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # テーブル行を追加
        for i, (_, row) in enumerate(data.iterrows()):
            rank = i + 1  # 各ランキング内での順位
            
            # 基本情報を取得
            basic_info = get_stock_basic_info(engine, row['symbol'])
            
            # テクニカルデータを取得
            technical_data = get_stock_technical_data(engine, row['symbol'], days_back=30)
            
            # 投資判断を生成
            recommendation = generate_investment_recommendation(row, technical_data, basic_info)
            
            # 推奨アクションに基づく背景色
            bg_color = get_recommendation_color(recommendation['action'])
            
            # ウォッチリスト用メタデータ
            ranking_type = row.get('ranking_type', 'unknown')
            watchlist_metadata = {
                'price': basic_info.get('current_price', 0),
                'rsi': technical_data.get('rsi_14', [0]).iloc[-1] if not technical_data.empty else 0,
                'score': row['total_score'],
                'analysis_date': report_date.isoformat() if hasattr(report_date, 'isoformat') else str(report_date),
                'rank': rank,
                'market_type': row.get('market_type', 'Unknown'),
                'ranking_type': ranking_type
            }
            
            # チェックボックスを生成
            checkbox_html = generate_simple_watchlist_checkbox(row['symbol'], 'top_score_stocks', watchlist_metadata)
            
            table_html += f"""
                        <tr style="background-color: {bg_color}; color: #333;">
                            <td>{checkbox_html}</td>
                            <td>{rank}</td>
                            <td><strong>{row['symbol']}</strong><br><small>{basic_info.get('company_name', 'N/A')}</small></td>
                            <td><strong>{row['total_score']:.1f}</strong></td>
                            <td>{row['value_score']:.1f}</td>
                            <td>{row['growth_score']:.1f}</td>
                            <td>{row['quality_score']:.1f}</td>
                            <td>{row['momentum_score']:.1f}</td>
                            <td>{row['macro_sector_score']:.1f}</td>
                            <td><strong>{recommendation['action']}</strong></td>
                            <td>
                                <button class="chart-button" onclick="document.getElementById('chart-{row['symbol']}-{ranking_type}').scrollIntoView({{behavior: 'smooth'}})">
                                    📊 チャート
                                </button>
                            </td>
                        </tr>
            """
        
        table_html += """
                    </tbody>
                </table>
        """
        return table_html
    
    # 1. 米国株全銘柄ランキング
    if not us_top_stocks_all.empty:
        html_content += create_score_ranking_table(us_top_stocks_all, "米国株スコア上位ランキング（全銘柄）", "🇺🇸", "全銘柄から選出")
    
    # 2. 米国株良質銘柄ランキング
    if not us_top_stocks_filtered.empty:
        html_content += create_score_ranking_table(us_top_stocks_filtered, "米国株スコア上位ランキング（良質銘柄）", "🇺🇸✨", "フィルター適用済み")
    
    # 3. 日本株全銘柄ランキング
    if not jp_top_stocks_all.empty:
        html_content += create_score_ranking_table(jp_top_stocks_all, "日本株スコア上位ランキング（全銘柄）", "🇯🇵", "全銘柄から選出")
    
    # 4. 日本株良質銘柄ランキング
    if not jp_top_stocks_filtered.empty:
        html_content += create_score_ranking_table(jp_top_stocks_filtered, "日本株スコア上位ランキング（良質銘柄）", "🇯🇵✨", "フィルター適用済み")
    
    html_content += """
            <h2>📈 詳細分析</h2>
    """
    
    # 各銘柄の詳細分析
    for i, (_, row) in enumerate(top_stocks.iterrows()):
        symbol = row['symbol']
        
        # 🔧 修正: 各種データを取得（エラーハンドリング追加）
        basic_info = get_stock_basic_info(engine, symbol)
        
        try:
            fundamental_data = get_stock_fundamental_data(engine, symbol, years_back=5)  # 5年分に修正
        except Exception as e:
            print(f"ファンダメンタルデータ取得エラー {symbol}: {e}")
            fundamental_data = pd.DataFrame()
        
        try:
            technical_data = get_stock_technical_data(engine, symbol, days_back=90)  # 3ヶ月分に修正
        except Exception as e:
            print(f"テクニカルデータ取得エラー {symbol}: {e}")
            technical_data = pd.DataFrame()
        
        # 🔧 デバッグ: technical_dataの中身を確認
        if not technical_data.empty:
            print(f"=== {symbol} テクニカルデータ確認 ===")
            print(f"データ件数: {len(technical_data)}")
            print(f"カラム: {list(technical_data.columns)}")
            if 'close' in technical_data.columns:
                print(f"Close価格範囲: {technical_data['close'].min():.2f} - {technical_data['close'].max():.2f}")
                print(f"Close価格サンプル (最新5件):")
                print(technical_data[['date', 'close']].tail().to_string())
            else:
                print("⚠️ 'close' カラムが見つかりません")
            print("=" * 50)
        else:
            print(f"⚠️ {symbol}: technical_dataが空です")
        
        try:
            weekly_data = get_stock_weekly_data(engine, symbol, weeks_back=52)  # 1年分に修正
        except Exception as e:
            print(f"週次データ取得エラー {symbol}: {e}")
            weekly_data = pd.DataFrame()
        
        try:
            financial_metrics = get_stock_financial_metrics(engine, symbol, years_back=5)  # 5年分に修正
        except Exception as e:
            print(f"財務指標取得エラー {symbol}: {e}")
            financial_metrics = pd.DataFrame()
        
        try:
            sector_data = get_sector_comparison_data(engine, symbol, days_back=90)  # 3ヶ月分に修正
        except Exception as e:
            print(f"セクター比較データ取得エラー {symbol}: {e}")
            sector_data = pd.DataFrame()
        
        try:
            score_history = get_stock_score_history(engine, symbol, days_back=252)  # 1年分
        except Exception as e:
            print(f"スコア履歴取得エラー {symbol}: {e}")
            score_history = pd.DataFrame()
        
        # スコア分析
        score_analysis = analyze_score_components(row)
        
        # 投資判断
        recommendation = generate_investment_recommendation(row, technical_data, basic_info)
        
        # 推奨アクションに応じたクラス設定
        rec_class = ""
        if "買い" in recommendation['action']:
            rec_class = "recommendation"
        elif "推奨" in recommendation['action']:
            rec_class = "recommendation recommend"
        elif "様子見" in recommendation['action']:
            rec_class = "recommendation hold"
        else:
            rec_class = "recommendation sell"
        
        # 基本情報の値を事前に計算
        current_price_str = f"{basic_info.get('current_price', 0):.2f}" if basic_info.get('current_price', 0) > 0 else 'N/A'
        
        # 時価総額の適切な表示（ドル単位で格納されているデータを適切な単位で表示）
        market_cap_raw = basic_info.get('market_cap', 0)
        if market_cap_raw >= 1_000_000_000_000:  # 1兆ドル以上
            market_cap_str = f"{market_cap_raw / 1_000_000_000_000:.1f}T"
        elif market_cap_raw >= 1_000_000_000:  # 10億ドル以上
            market_cap_str = f"{market_cap_raw / 1_000_000_000:.1f}B"
        elif market_cap_raw >= 1_000_000:  # 100万ドル以上
            market_cap_str = f"{market_cap_raw / 1_000_000:.1f}M"
        elif market_cap_raw >= 1_000:  # 1000ドル以上
            market_cap_str = f"{market_cap_raw / 1_000:.1f}K"
        elif market_cap_raw > 0:
            market_cap_str = f"{market_cap_raw:.0f}"
        else:
            market_cap_str = "N/A"
        
        per_str = f"{basic_info.get('per', 0):.1f}" if basic_info.get('per', 0) > 0 else 'N/A'
        pbr_str = f"{basic_info.get('pbr', 0):.1f}" if basic_info.get('pbr', 0) > 0 else 'N/A'
        
        # ROE/ROICは小数形式で格納されているため100倍してパーセンテージ表示
        roe_value = basic_info.get('roe', 0)
        roic_value = basic_info.get('roic', 0)
        
        # ROEは常に小数形式（0.25 = 25%）
        roe_str = f"{roe_value * 100:.1f}%" if roe_value > 0 else 'N/A'
        
        # ROICは値の大きさで判断（1以下なら小数形式、1超なら既にパーセント形式）
        if roic_value > 0:
            if roic_value <= 1:
                roic_str = f"{roic_value * 100:.1f}%"  # 小数形式の場合
            else:
                roic_str = f"{roic_value:.1f}%"  # 既にパーセント形式の場合
        else:
            roic_str = 'N/A'
        
        # 🔧 修正: 詳細チャートを作成（エラーハンドリング改善）
        chart_html = ""
        try:
            # 🔧 修正: 株価データを別途抽出
            stock_price_data = pd.DataFrame()
            if not technical_data.empty:
                # 株価関連のカラムのみを抽出
                price_columns = ['date', 'close']
                if 'sma_20' in technical_data.columns:
                    price_columns.append('sma_20')
                if 'sma_40' in technical_data.columns:
                    price_columns.append('sma_40')
                
                # 存在するカラムのみを選択
                available_columns = [col for col in price_columns if col in technical_data.columns]
                if available_columns:
                    stock_price_data = technical_data[available_columns].copy()
                    print(f"株価データ抽出完了 {symbol}: {available_columns}")
                else:
                    print(f"⚠️ 株価データが見つかりません {symbol}")
            
            # すべての必要な変数が定義されていることを確認
            print(f"スコア上位銘柄 - チャート作成開始 {symbol}")
            chart = create_enhanced_stock_detail_chart(
                symbol=symbol, 
                stock_data=stock_price_data,  # 🔧 修正: 株価データのみを渡す
                score_data=row, 
                weekly_data=weekly_data, 
                financial_metrics=financial_metrics,
                sector_comparison=sector_data, 
                technical_data=technical_data,  # テクニカル指標用として別途渡す
                basic_info=basic_info, 
                score_history=score_history, 
                fundamental_data=fundamental_data,
                engine=engine
            )
            chart_html = chart.to_html(full_html=False, include_plotlyjs=False)
            print(f"スコア上位銘柄 - 詳細チャート作成成功 {symbol}")
        except Exception as e:
            print(f"スコア上位銘柄 - 詳細チャート作成エラー {symbol}: {e}")
            # フォールバック: 基本的なチャートを作成
            try:
                print(f"スコア上位銘柄 - フォールバックチャート作成開始 {symbol}")
                chart = create_basic_fallback_chart(
                    symbol=symbol, 
                    score_data=row, 
                    fundamental_data=fundamental_data, 
                    technical_data=technical_data, 
                    basic_info=basic_info
                )
                chart_html = chart.to_html(full_html=False, include_plotlyjs=False)
                print(f"スコア上位銘柄 - フォールバックチャート作成成功 {symbol}")
            except Exception as e2:
                print(f"スコア上位銘柄 - フォールバックチャート作成エラー {symbol}: {e2}")
                # 🔧 修正: エラー時はシンプルなメッセージを表示
                chart_html = f"""
                <div style="padding: 20px; text-align: center; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;">
                    <h4>📊 {symbol} - チャート生成中</h4>
                    <p>現在価格: {current_price_str} | 総合スコア: {row['total_score']:.1f}</p>
                    <p style="color: #6c757d;">詳細チャートは次回の更新で表示予定です。</p>
                </div>
                """
        
        # 詳細セクション用のウォッチリストメタデータ
        detail_ranking_type = row.get('ranking_type', 'unknown')
        detail_market_type = row.get('market_type', 'Unknown')
        
        # グローバルランキング取得
        target_date_str = str(report_date.date() if hasattr(report_date, 'date') else report_date)
        global_rank = get_market_global_ranking(engine, symbol, detail_market_type, target_date_str)
        
        detail_watchlist_metadata = {
            'price': basic_info.get('current_price', 0),
            'rsi': technical_data.get('rsi_14', [0]).iloc[-1] if not technical_data.empty else 0,
            'score': row['total_score'],
            'analysis_date': report_date.isoformat() if hasattr(report_date, 'isoformat') else str(report_date),
            'rank': i + 1,
            'global_rank': global_rank['rank'],
            'total_market_stocks': global_rank['total_stocks'],
            'ranking_type': detail_ranking_type,
            'market_type': detail_market_type
        }
        
        # 詳細セクション用チェックボックス
        detail_checkbox = generate_simple_watchlist_checkbox(symbol, 'top_score_stocks', detail_watchlist_metadata)
        
        # ランキング表示文字列作成
        rank_display = f"#{i+1}"  # カテゴリ内順位
        if global_rank['rank'] > 0:
            rank_display += f" ({global_rank['rank']}/{global_rank['total_stocks']}stocks)"
        
        html_content += f"""
            <div class="stock-card">
                <div class="stock-header">
                    <div class="stock-title">
                        {rank_display} {symbol} {detail_checkbox}
                        <div style="font-size: 0.8em; color: #2980b9; margin-top: 5px;">
                            {basic_info.get('company_name', 'N/A')}
                        </div>
                        <div style="font-size: 0.7em; color: #666;">
                            {basic_info.get('industry', 'N/A')} | {basic_info.get('sector', 'N/A')}
                        </div>
                        <div style="font-size: 0.6em; color: #888; margin-top: 3px; padding: 2px 6px; background-color: #f0f0f0; border-radius: 3px; display: inline-block;">
                            {'✨ 良質銘柄' if detail_ranking_type == 'filtered' else '📊 全銘柄'} | {detail_market_type}市場
                        </div>
                    </div>
                    <div class="total-score">{row['total_score']:.1f}点</div>
                </div>
                
                <!-- 基本情報 -->
                <div class="basic-info">
                    <div class="info-item">
                        <span>現在株価:</span>
                        <span><strong>{current_price_str}</strong></span>
                    </div>
                    <div class="info-item">
                        <span>時価総額:</span>
                        <span>{market_cap_str}</span>
                    </div>
                    <div class="info-item">
                        <span>PER:</span>
                        <span>{per_str}</span>
                    </div>
                    <div class="info-item">
                        <span>PBR:</span>
                        <span>{pbr_str}</span>
                    </div>
                    <div class="info-item">
                        <span>ROE:</span>
                        <span>{roe_str}</span>
                    </div>
                    <div class="info-item">
                        <span>ROIC:</span>
                        <span>{roic_str}</span>
                    </div>
                </div>
                
                <!-- 分析結果 -->
                <div class="analysis-section">
                    <h4>💪 強み・弱み分析</h4>
                    <p><span class="strengths">強み:</span> {', '.join(score_analysis['strengths']) if score_analysis['strengths'] else 'なし'}</p>
                    <p><span class="weaknesses">弱み:</span> {', '.join(score_analysis['weaknesses']) if score_analysis['weaknesses'] else 'なし'}</p>
                    
                    <h4>📊 カテゴリ別評価</h4>
                    <ul>
                        <li><strong>割安性:</strong> {score_analysis['value_analysis']}</li>
                        <li><strong>成長性:</strong> {score_analysis['growth_analysis']}</li>
                        <li><strong>財務品質:</strong> {score_analysis['quality_analysis']}</li>
                        <li><strong>モメンタム:</strong> {score_analysis['momentum_analysis']}</li>
                        <li><strong>マクロ環境:</strong> {score_analysis['macro_analysis']}</li>
                    </ul>
                </div>
                
                <!-- 投資判断 -->
                <div class="{rec_class}">
                    <h4>🎯 投資判断: {recommendation['action']}</h4>
                    <p><strong>判断理由:</strong> {recommendation['reasoning']}</p>
                    <p><strong>リスクレベル:</strong> {recommendation['risk_level']}</p>
                    <p><strong>投資期間:</strong> {recommendation['time_horizon']}</p>
                    <p><strong>エントリー戦略:</strong> {recommendation['entry_strategy']}</p>
                    <p><strong>出口戦略:</strong> {recommendation['exit_strategy']}</p>
                </div>
                
                <!-- チャート -->
                <div class="chart-container" id="chart-{symbol}-{detail_ranking_type}">
                    <div style="margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;">
                        <button class="detailed-report-btn" onclick="generateDetailedReport('{symbol}')">
                            📊 詳細レポート生成
                        </button>
                        <button class="back-to-ranking-btn" onclick="scrollToTop()">
                            🔙 ランキングに戻る
                        </button>
                    </div>
                    {chart_html}
                </div>
            </div>
        """
    
    # HTMLの終了部分
    html_content += f"""
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
            <p style="color: #666; font-size: 0.9em;">
                ⚠️ 本レポートは投資判断の参考情報です。投資は自己責任で行ってください。<br>
                スコアは過去データに基づく分析であり、将来の投資成果を保証するものではありません。
            </p>
        </div>
        
        {add_simple_watchlist_javascript()}
        
        <script>
            // ランキングテーブルへ戻るスクロール機能
            function scrollToTop() {{
                window.scrollTo({{
                    top: 0,
                    behavior: 'smooth'
                }});
            }}
            
            // 詳細レポート生成機能
            function generateDetailedReport(symbol) {{
                console.log(`🚀 詳細レポート生成開始: ${{symbol}}`);
                
                // ボタンを無効化
                const button = event.target;
                const originalText = button.innerHTML;
                button.disabled = true;
                button.innerHTML = '⏳ 生成中...';
                
                // APIエンドポイントに詳細レポート生成を要求
                fetch('http://127.0.0.1:5001/api/generate_detailed_report', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        symbol: symbol
                    }})
                }})
                .then(response => {{
                    if (!response.ok) {{
                        throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
                    }}
                    return response.text().then(text => {{
                        try {{
                            return JSON.parse(text);
                        }} catch (parseError) {{
                            console.error('❌ JSON解析エラー:', parseError);
                            console.error('❌ レスポンス:', text);
                            throw new Error(`Response parsing failed: ${{parseError.message}}`);
                        }}
                    }});
                }})
                .then(data => {{
                    console.log('📊 API応答:', data);
                    
                    if (data && data.success) {{
                        console.log(`✅ ${{symbol}} の詳細レポート生成完了`);
                        
                        // ボタンを「レポートを開く」に変更
                        button.innerHTML = '📋 レポートを開く';
                        button.disabled = false;
                        button.style.background = 'linear-gradient(135deg, #27ae60 0%, #229954 100%)';
                        
                        // 成功メッセージを表示（ポップアップブロック対応版）
                        showTemporaryMessage(`✅ ${{symbol}} の詳細レポート生成完了！ボタンをクリックして開いてください`, 'success', 5000);
                        
                        // ボタンクリックでレポートを開く（ポップアップブロッカー回避）
                        button.onclick = function() {{
                            console.log(`🔗 レポートを開きます: ${{data.report_url}}`);
                            window.open(data.report_url, '_blank');
                            
                            // ボタンを元に戻す
                            setTimeout(() => {{
                                button.innerHTML = originalText;
                                button.style.background = '';
                                button.onclick = () => generateDetailedReport(symbol);
                            }}, 2000);
                        }};
                        
                    }} else {{
                        const errorMsg = data && data.error ? data.error : '不明なエラー';
                        console.error('❌ 詳細レポート生成失敗:', errorMsg);
                        showTemporaryMessage(`❌ ${{symbol}} の詳細レポート生成に失敗: ${{errorMsg}}`, 'error');
                        
                        // ボタンを元に戻す
                        button.innerHTML = originalText;
                        button.disabled = false;
                    }}
                }})
                .catch(error => {{
                    console.error('🚫 API呼び出しエラー:', error);
                    
                    let errorMessage = '❌ 詳細レポート生成でエラーが発生しました';
                    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {{
                        errorMessage = '⚠️ APIサーバーに接続できません。サーバーを起動してください: python start_watchlist_api.py';
                    }} else if (error.message.includes('timeout')) {{
                        errorMessage = '⚠️ レポート生成がタイムアウトしました。しばらく待ってから再試行してください';
                    }}
                    
                    showTemporaryMessage(errorMessage, 'error');
                    
                    // エラー時もボタンを元に戻す
                    button.innerHTML = originalText;
                    button.disabled = false;
                    button.style.background = '';
                }});
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content 


def get_stock_score_history(engine, symbol: str, days_back: Optional[int] = 365) -> pd.DataFrame:
    """
    銘柄のスコア履歴を取得（デフォルト1年）
    
    Args:
        engine: データベースエンジン
        symbol: 銘柄コード
        days_back: 何日分のデータを取得するか（デフォルト365日=1年）
        
    Returns:
        スコア履歴データ
    """
    start_date = None
    if days_back is not None:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    query = text("""
    SELECT 
        date,
        total_score,
        value_score,
        growth_score,
        quality_score,
        momentum_score,
        macro_sector_score
    FROM backtest_results.daily_scores
    WHERE symbol = :symbol
      AND (:start_date IS NULL OR date >= :start_date)
    ORDER BY date
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"symbol": symbol, "start_date": start_date})
        return df
    except Exception as e:
        print(f"スコア履歴取得エラー {symbol}: {e}")
        return pd.DataFrame()


def generate_rsi35_below_report(engine, target_date: str = None) -> str:
    """
    RSI35以下の高スコア銘柄レポートのHTMLを生成（ウォッチリスト機能付き）
    
    Args:
        engine: データベースエンジン
        target_date: 対象日
        
    Returns:
        HTMLコンテンツ
    """
    if target_date is None:
        # 最新の日付を取得
        query = "SELECT MAX(date) as max_date FROM backtest_results.daily_scores"
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            report_date = result.max_date
    else:
        report_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    # RSI35以下の高スコア銘柄を取得
    rsi35_query = text("""
    SELECT DISTINCT
        ds.symbol,
        ds.date,
        ds.total_score,
        ds.value_score,
        ds.growth_score,
        ds.quality_score,
        ds.momentum_score,
        ds.macro_sector_score,
        ti.rsi_14
    FROM backtest_results.daily_scores ds
    INNER JOIN calculated_metrics.technical_indicators ti ON ds.symbol = ti.symbol AND ds.date = ti.date
    WHERE ds.date = :target_date
    AND ti.rsi_14 <= 35
    AND ti.rsi_14 > 0
    ORDER BY ds.total_score DESC
    LIMIT 20
    """)
    
    try:
        rsi35_stocks = pd.read_sql(rsi35_query, engine, params={"target_date": report_date})
    except Exception as e:
        print(f"RSI35以下データ取得エラー: {e}")
        return f"<html><body><h1>データ取得エラー: {e}</h1></body></html>"
    
    if rsi35_stocks.empty:
        return f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="utf-8">
            <title>RSI35以下 買い候補分析レポート</title>
            <script src="https://cdn.plot.ly/plotly-2.29.0.min.js"></script>
            {add_simple_watchlist_css()}
            <style>
                body {{
                    font-family: system-ui, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 2px solid #e74c3c;
                    padding-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📉 RSI35以下 買い候補分析レポート</h1>
                    <p>分析日: {report_date}</p>
                    <p style="color: #e74c3c; font-weight: bold;">
                        🔍 該当する銘柄がありませんでした
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
    
    # HTMLの開始部分
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>RSI35以下 買い候補分析レポート</title>
        <script src="https://cdn.plot.ly/plotly-2.29.0.min.js"></script>
        {add_simple_watchlist_css()}
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                line-height: 1.5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #e74c3c;
                padding-bottom: 20px;
            }}
            .stock-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 20px 0;
                padding: 20px;
                background-color: #fff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .stock-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            .stock-title {{
                font-size: 1.5em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .total-score {{
                font-size: 2em;
                font-weight: bold;
                color: #e74c3c;
            }}
            .analysis-section {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 6px;
            }}
            .strengths {{
                color: #27ae60;
                font-weight: bold;
            }}
            .weaknesses {{
                color: #e74c3c;
                font-weight: bold;
            }}
            .recommendation {{
                background-color: #e8f5e8;
                border-left: 4px solid #27ae60;
                padding: 15px;
                margin: 15px 0;
            }}
            .recommendation.recommend {{
                background-color: #e3f2fd;
                border-left-color: #2196f3;
            }}
            .recommendation.hold {{
                background-color: #fff3cd;
                border-left-color: #ffc107;
            }}
            .recommendation.sell {{
                background-color: #f8d7da;
                border-left-color: #dc3545;
            }}
            .basic-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin: 15px 0;
                padding: 15px;
                background-color: #e3f2fd;
                border-radius: 6px;
            }}
            .info-item {{
                display: flex;
                justify-content: space-between;
            }}
            .chart-container {{
                margin: 20px 0;
                height: 2300px;
                overflow: visible;
            }}
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .summary-table th,
            .summary-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            .summary-table th {{
                background-color: #3498db;
                color: white;
            }}
            .rank-1 {{ background-color: #ffd700; }}
            .rank-2 {{ background-color: #c0c0c0; }}
            .rank-3 {{ background-color: #cd7f32; }}
            .rsi-highlight {{
                background-color: #ffebee;
                font-weight: bold;
                color: #c62828;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📉 RSI35以下 買い候補分析レポート</h1>
                <p>分析日: {report_date} | 売られすぎ銘柄{len(rsi35_stocks)}銘柄</p>
                <p style="color: #e74c3c; font-weight: bold;">
                    🔍 RSI35以下の売られすぎ銘柄をスコア順で表示
                </p>
            </div>
            
            <!-- サマリーテーブル -->
            <h2>🏆 ランキング一覧</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Watch</th>
                        <th>順位</th>
                        <th>銘柄</th>
                        <th>RSI</th>
                        <th>総合スコア</th>
                        <th>Value</th>
                        <th>Growth</th>
                        <th>Quality</th>
                        <th>Momentum</th>
                        <th>Macro</th>
                        <th>推奨アクション</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # サマリーテーブルの行を追加
    for i, (_, row) in enumerate(rsi35_stocks.iterrows()):
        rank_class = f"rank-{i+1}" if i < 3 else ""
        
        # 基本情報を取得
        basic_info = get_stock_basic_info(engine, row['symbol'])
        
        # テクニカルデータを取得
        technical_data = get_stock_technical_data(engine, row['symbol'], days_back=30)
        
        # 投資判断を生成
        recommendation = generate_investment_recommendation(row, technical_data, basic_info)
        
        # ウォッチリスト用メタデータ
        watchlist_metadata = {
            'price': basic_info.get('current_price', 0),
            'rsi': row['rsi_14'],
            'score': row['total_score'],
            'analysis_date': report_date.isoformat() if hasattr(report_date, 'isoformat') else str(report_date),
            'rank': i + 1,
            'growth_score': row['growth_score']
        }
        
        # チェックボックスを生成（analysis_typeをrsi35_belowに統一）
        checkbox_html = generate_simple_watchlist_checkbox(row['symbol'], 'rsi35_below', watchlist_metadata)
        
        html_content += f"""
                    <tr class="{rank_class}">
                        <td>{checkbox_html}</td>
                        <td>{i+1}</td>
                        <td><strong>{row['symbol']}</strong><br><small>{basic_info.get('company_name', 'N/A')}</small></td>
                        <td class="rsi-highlight">{row['rsi_14']:.1f}</td>
                        <td><strong>{row['total_score']:.1f}</strong></td>
                        <td>{row['value_score']:.1f}</td>
                        <td>{row['growth_score']:.1f}</td>
                        <td>{row['quality_score']:.1f}</td>
                        <td>{row['momentum_score']:.1f}</td>
                        <td>{row['macro_sector_score']:.1f}</td>
                        <td>{recommendation['action']}</td>
                    </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
                
                <h2>📈 詳細分析</h2>
    """
    
    # 各銘柄の詳細分析
    for i, (_, row) in enumerate(rsi35_stocks.iterrows()):
        symbol = row['symbol']
        
        # ⚡ 軽量化：必要最小限のデータのみ取得
        basic_info = get_stock_basic_info(engine, symbol)
        
        # 修正：テクニカルデータは3ヶ月分
        technical_data = get_stock_technical_data(engine, symbol, days_back=90)
        
        # 🔧 修正: RSI35以下レポートで不足していたデータを取得
        try:
            weekly_data = get_stock_weekly_data(engine, symbol, weeks_back=52)  # 1年分に修正
        except Exception as e:
            print(f"週次データ取得エラー {symbol}: {e}")
            weekly_data = pd.DataFrame()
        
        try:
            financial_metrics = get_stock_financial_metrics(engine, symbol, years_back=5)  # 5年分に修正
        except Exception as e:
            print(f"財務指標取得エラー {symbol}: {e}")
            financial_metrics = pd.DataFrame()
        
        try:
            sector_data = get_sector_comparison_data(engine, symbol, days_back=90)  # 3ヶ月分に修正
        except Exception as e:
            print(f"セクター比較データ取得エラー {symbol}: {e}")
            sector_data = pd.DataFrame()
        
        try:
            score_history = get_stock_score_history(engine, symbol, days_back=365)  # 1年分に修正
        except Exception as e:
            print(f"スコア履歴取得エラー {symbol}: {e}")
            score_history = pd.DataFrame()
        
        try:
            fundamental_data = get_stock_fundamental_data(engine, symbol, years_back=5)  # 5年分に修正
        except Exception as e:
            print(f"ファンダメンタルデータ取得エラー {symbol}: {e}")
            fundamental_data = pd.DataFrame()
        
        # ⚡ スコア分析は既存データから生成（DB未使用）
        score_analysis = analyze_score_components(row)
        
        # ⚡ 軽量化：シンプルな投資判断生成
        if row['rsi_14'] < 25 and row['total_score'] > 7:
            recommendation = {
                'action': '強い買い推奨',
                'reasoning': f'RSI {row["rsi_14"]:.1f}の強い売られすぎと高スコア{row["total_score"]:.1f}の組み合わせ',
                'risk_level': '中',
                'time_horizon': '短期〜中期',
                'entry_strategy': '分割買い推奨',
                'exit_strategy': 'RSI50超えで利確検討'
            }
        elif row['rsi_14'] < 30 and row['total_score'] > 5:
            recommendation = {
                'action': '買い推奨',
                'reasoning': f'RSI {row["rsi_14"]:.1f}の売られすぎ状態、スコア{row["total_score"]:.1f}',
                'risk_level': '中',
                'time_horizon': '中期',
                'entry_strategy': '段階的エントリー',
                'exit_strategy': 'テクニカル反転で利確'
            }
        else:
            recommendation = {
                'action': '様子見',
                'reasoning': f'RSI {row["rsi_14"]:.1f}、追加的な売り圧力の可能性',
                'risk_level': '高',
                'time_horizon': '待機',
                'entry_strategy': '更なる下落を待つ',
                'exit_strategy': '該当なし'
            }
        
        # 推奨アクションに応じたクラス設定
        rec_class = ""
        if "買い" in recommendation['action']:
            rec_class = "recommendation"
        elif "推奨" in recommendation['action']:
            rec_class = "recommendation recommend"
        elif "様子見" in recommendation['action']:
            rec_class = "recommendation hold"
        else:
            rec_class = "recommendation sell"
        
        # 基本情報の値を事前に計算
        current_price_str = f"{basic_info.get('current_price', 0):.2f}" if basic_info.get('current_price', 0) > 0 else 'N/A'
        
        # 時価総額の適切な表示
        market_cap_raw = basic_info.get('market_cap', 0)
        if market_cap_raw >= 1_000_000_000_000:
            market_cap_str = f"{market_cap_raw / 1_000_000_000_000:.1f}T"
        elif market_cap_raw >= 1_000_000_000:
            market_cap_str = f"{market_cap_raw / 1_000_000_000:.1f}B"
        elif market_cap_raw >= 1_000_000:
            market_cap_str = f"{market_cap_raw / 1_000_000:.1f}M"
        elif market_cap_raw >= 1_000:
            market_cap_str = f"{market_cap_raw / 1_000:.1f}K"
        elif market_cap_raw > 0:
            market_cap_str = f"{market_cap_raw:.0f}"
        else:
            market_cap_str = "N/A"
        
        per_str = f"{basic_info.get('per', 0):.1f}" if basic_info.get('per', 0) > 0 else 'N/A'
        pbr_str = f"{basic_info.get('pbr', 0):.1f}" if basic_info.get('pbr', 0) > 0 else 'N/A'
        
        # ROE/ROICの計算
        roe_value = basic_info.get('roe', 0)
        roic_value = basic_info.get('roic', 0)
        
        roe_str = f"{roe_value * 100:.1f}%" if roe_value > 0 else 'N/A'
        
        if roic_value > 0:
            if roic_value <= 1:
                roic_str = f"{roic_value * 100:.1f}%"
            else:
                roic_str = f"{roic_value:.1f}%"
        else:
            roic_str = 'N/A'
        
        # 🔧 修正: 詳細チャートを作成（エラーハンドリング改善）
        chart_html = ""
        try:
            # すべての必要な変数が定義されていることを確認
            print(f"チャート作成開始 {symbol}")
            chart = create_enhanced_stock_detail_chart(
                symbol=symbol, 
                stock_data=technical_data,  # technical_dataをstock_dataとして使用
                score_data=row, 
                weekly_data=weekly_data, 
                financial_metrics=financial_metrics,
                sector_comparison=sector_data, 
                technical_data=technical_data, 
                basic_info=basic_info, 
                score_history=score_history, 
                fundamental_data=fundamental_data,
                engine=engine
            )
            chart_html = chart.to_html(full_html=False, include_plotlyjs=False)
            print(f"詳細チャート作成成功 {symbol}")
        except Exception as e:
            print(f"詳細チャート作成エラー {symbol}: {e}")
            try:
                # フォールバックチャート作成
                print(f"フォールバックチャート作成開始 {symbol}")
                chart = create_basic_fallback_chart(
                    symbol=symbol, 
                    score_data=row, 
                    fundamental_data=fundamental_data, 
                    technical_data=technical_data, 
                    basic_info=basic_info
                )
                chart_html = chart.to_html(full_html=False, include_plotlyjs=False)
                print(f"フォールバックチャート作成成功 {symbol}")
            except Exception as e2:
                print(f"フォールバックチャート作成エラー {symbol}: {e2}")
                # 🔧 修正: エラー時はシンプルなメッセージを表示
                chart_html = f"""
                <div style="padding: 20px; text-align: center; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;">
                    <h4>📊 {symbol} - チャート生成中</h4>
                    <p>現在価格: {current_price_str} | RSI: {row['rsi_14']:.1f} | スコア: {row['total_score']:.1f}</p>
                    <p style="color: #6c757d;">詳細チャートは次回の更新で表示予定です。</p>
                </div>
                """
        
        # 詳細セクション用のウォッチリストメタデータ
        detail_watchlist_metadata = {
            'price': basic_info.get('current_price', 0),
            'rsi': row['rsi_14'],
            'score': row['total_score'],
            'analysis_date': report_date.isoformat() if hasattr(report_date, 'isoformat') else str(report_date),
            'rank': i + 1,
            'growth_score': row['growth_score']
        }
        
        # 詳細セクション用チェックボックス
        detail_checkbox = generate_simple_watchlist_checkbox(symbol, 'rsi35_below', detail_watchlist_metadata)
        
        html_content += f"""
            <div class="stock-card">
                <div class="stock-header">
                    <div class="stock-title">
                        #{i+1} {symbol} {detail_checkbox}
                        <div style="font-size: 0.8em; color: #2980b9; margin-top: 5px;">
                            {basic_info.get('company_name', 'N/A')}
                        </div>
                        <div style="font-size: 0.7em; color: #666;">
                            {basic_info.get('industry', 'N/A')} | {basic_info.get('sector', 'N/A')}
                        </div>
                        <div style="font-size: 0.9em; color: #c62828; font-weight: bold; margin-top: 5px;">
                            📉 RSI: {row['rsi_14']:.1f} (売られすぎ)
                        </div>
                    </div>
                    <div class="total-score">{row['total_score']:.1f}点</div>
                </div>
                
                <!-- 基本情報 -->
                <div class="basic-info">
                    <div class="info-item">
                        <span>現在株価:</span>
                        <span><strong>{current_price_str}</strong></span>
                    </div>
                    <div class="info-item">
                        <span>時価総額:</span>
                        <span>{market_cap_str}</span>
                    </div>
                    <div class="info-item">
                        <span>PER:</span>
                        <span>{per_str}</span>
                    </div>
                    <div class="info-item">
                        <span>PBR:</span>
                        <span>{pbr_str}</span>
                    </div>
                    <div class="info-item">
                        <span>ROE:</span>
                        <span>{roe_str}</span>
                    </div>
                    <div class="info-item">
                        <span>ROIC:</span>
                        <span>{roic_str}</span>
                    </div>
                </div>
                
                <!-- 分析結果 -->
                <div class="analysis-section">
                    <h4>💪 強み・弱み分析</h4>
                    <p><span class="strengths">強み:</span> {', '.join(score_analysis['strengths']) if score_analysis['strengths'] else 'なし'}</p>
                    <p><span class="weaknesses">弱み:</span> {', '.join(score_analysis['weaknesses']) if score_analysis['weaknesses'] else 'なし'}</p>
                    
                    <h4>📊 カテゴリ別評価</h4>
                    <ul>
                        <li><strong>割安性:</strong> {score_analysis['value_analysis']}</li>
                        <li><strong>成長性:</strong> {score_analysis['growth_analysis']}</li>
                        <li><strong>財務品質:</strong> {score_analysis['quality_analysis']}</li>
                        <li><strong>モメンタム:</strong> {score_analysis['momentum_analysis']}</li>
                        <li><strong>マクロ環境:</strong> {score_analysis['macro_analysis']}</li>
                    </ul>
                    
                    <h4>📉 RSI分析</h4>
                    <p><strong>売られすぎ度:</strong> RSI {row['rsi_14']:.1f} - 短期的な反発の可能性あり</p>
                    <p><strong>リバウンド期待:</strong> 技術的には買い場の可能性が高い水準</p>
                </div>
                
                <!-- 投資判断 -->
                <div class="{rec_class}">
                    <h4>🎯 投資判断: {recommendation['action']}</h4>
                    <p><strong>判断理由:</strong> {recommendation['reasoning']}</p>
                    <p><strong>リスクレベル:</strong> {recommendation['risk_level']}</p>
                    <p><strong>投資期間:</strong> {recommendation['time_horizon']}</p>
                    <p><strong>エントリー戦略:</strong> {recommendation['entry_strategy']}</p>
                    <p><strong>出口戦略:</strong> {recommendation['exit_strategy']}</p>
                    <p><strong>RSI戦略:</strong> 売られすぎからの反発を狙った短期〜中期投資に適している可能性</p>
                </div>
                
                <!-- チャート -->
                <div class="chart-container">
                    {chart_html}
                </div>
            </div>
                        """
    
    # HTMLの終了部分
    html_content += """
            <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                <h3>📌 RSI35以下投資の注意事項</h3>
                <ul>
                    <li><strong>売られすぎ反発:</strong> RSI35以下は技術的に売られすぎを示唆しますが、さらなる下落リスクも存在します</li>
                    <li><strong>ファンダメンタル確認:</strong> 技術的指標だけでなく、企業の基本的価値も必ず確認してください</li>
                    <li><strong>分散投資:</strong> 一つの銘柄に集中せず、複数銘柄への分散投資を推奨します</li>
                    <li><strong>損切り設定:</strong> エントリー前に明確な損切りラインを設定してください</li>
                    <li><strong>市場環境:</strong> 全体相場の動向も投資判断に織り込んでください</li>
                </ul>
            </div>
        </div>
        {add_simple_watchlist_javascript()}
    </body>
    </html>
    """
    
    return html_content


def generate_rsi35_investment_recommendation(row: pd.Series, technical_data: pd.DataFrame, basic_info: Dict, rsi_value: float) -> Dict[str, str]:
    """
    RSI35以下銘柄専用の投資判断を生成
    
    Args:
        row: スコアデータの行
        technical_data: テクニカルデータ
        basic_info: 基本情報
        rsi_value: RSI値
        
    Returns:
        投資判断の辞書
    """
    total_score = row.get('total_score', 0)
    growth_score = row.get('growth_score', 0)
    value_score = row.get('value_score', 0)
    
    # RSI分析
    if rsi_value <= 20:
        rsi_analysis = f"RSI {rsi_value:.1f} - 極度の売られすぎ。強いリバウンドの可能性"
    elif rsi_value <= 30:
        rsi_analysis = f"RSI {rsi_value:.1f} - 売られすぎ水準。買いエントリー検討タイミング"
    else:
        rsi_analysis = f"RSI {rsi_value:.1f} - 売られすぎ水準に近い。慎重にエントリー検討"
    
    # スコアベース判断
    if total_score >= 55 and growth_score >= 8:
        action = "🟢 積極的買い推奨"
        reasoning = f"高スコア({total_score:.1f}点)かつ成長性良好。RSI売られすぎでエントリー好機"
        entry_strategy = "2-3回に分けて段階的エントリー。RSI反転確認後に追加購入"
        risk_management = "5%ストップロス設定。ポジションサイズは資金の3-5%以内"
        target_exit = "RSI70超過または20%利益で段階的利益確定"
    elif total_score >= 50:
        action = "🟡 慎重な買い検討"
        reasoning = f"中程度のスコア({total_score:.1f}点)。RSI売られすぎを活用した短期戦略"
        entry_strategy = "小ロットでテストエントリー。RSI底打ち確認後に追加"
        risk_management = "3-5%ストップロス。ポジションサイズは資金の2-3%以内"
        target_exit = "RSI60超過または15%利益で利益確定検討"
    elif total_score >= 45:
        action = "🟠 様子見・小ロット"
        reasoning = f"スコア({total_score:.1f}点)は平均的。RSI売られすぎだが慎重に"
        entry_strategy = "ごく小ロットでのエントリー。他の確認指標の好転待ち"
        risk_management = "2-3%ストップロス。ポジションサイズは資金の1-2%以内"
        target_exit = "RSI50超過または10%利益で早期利益確定"
    else:
        action = "🔴 エントリー非推奨"
        reasoning = f"スコア({total_score:.1f}点)が低い。RSI売られすぎでも基本面に懸念"
        entry_strategy = "エントリー見送り。スコア改善まで待機"
        risk_management = "投資対象外"
        target_exit = "投資対象外"
    
    return {
        'action': action,
        'rsi_analysis': rsi_analysis,
        'reasoning': reasoning,
        'entry_strategy': entry_strategy,
        'risk_management': risk_management,
        'target_exit': target_exit
    }
