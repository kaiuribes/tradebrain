# Enhanced core/orchestrator.py

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from agents.chart_agent import analyze_chart
from agents.sentiment_agent import analyze_sentiment
from agents.news_agent import fetch_news
from agents.llm_decider import decide_trade
from agents.llm_router import route_llm_task
from agents.risk_manager import apply_risk_rules
from agents.feedback_evaluator import update_strategy

CONFIG_PATH = "config/settings.json"

# Enhanced data structures
class DecisionType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"  
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingSignal:
    timestamp: datetime
    symbol: str
    chart_signal: str
    sentiment: str
    news_summary: List[str]
    llm_reasoning: str
    risk_assessment: str
    final_decision: DecisionType
    confidence_score: float
    metadata: Dict[str, Any]

class EnhancedOrchestrator:
    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.settings = self.load_settings()
        self.setup_logging()
        self.trading_history: List[TradingSignal] = []
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TradeBrain')
        
    def load_settings(self) -> Dict:
        """Load configuration with error handling"""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {self.config_path}")
            return self.get_default_config()
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON in config file")
            return self.get_default_config()
            
    def get_default_config(self) -> Dict:
        """Default configuration fallback"""
        return {
            "model": "mistral",
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "risk_tolerance": 0.02,
            "max_position_size": 1000,
            "trading_enabled": False
        }

    def analyze_multiple_timeframes(self, symbol: str) -> Dict[str, str]:
        """Enhanced chart analysis across multiple timeframes"""
        timeframes = {
            "short": {"period": "1d", "interval": "5m"},
            "medium": {"period": "5d", "interval": "1h"}, 
            "long": {"period": "1mo", "interval": "1d"}
        }
        
        signals = {}
        for tf_name, params in timeframes.items():
            try:
                signal = analyze_chart(
                    ticker=symbol,
                    period=params["period"],
                    interval=params["interval"]
                )
                signals[tf_name] = signal
                self.logger.info(f"{symbol} {tf_name}-term signal: {signal}")
            except Exception as e:
                self.logger.error(f"Chart analysis failed for {tf_name}: {e}")
                signals[tf_name] = "ERROR"
                
        return signals

    def enhanced_sentiment_analysis(self, news: List[str], symbol: str) -> Dict[str, Any]:
        """Enhanced sentiment with LLM reasoning"""
        try:
            # Basic sentiment from FinBERT
            base_sentiment = analyze_sentiment(news)
            
            # Enhanced reasoning via LLM
            news_text = " | ".join(news[:5])  # Limit for context
            llm_prompt = f"""
            Analyze the sentiment and market implications for {symbol}:
            News: {news_text}
            
            Provide:
            1. Overall sentiment (Bullish/Bearish/Neutral)
            2. Key catalysts mentioned
            3. Potential price impact (Low/Medium/High)
            4. Time horizon (Short/Medium/Long term)
            """
            
            llm_analysis = route_llm_task(llm_prompt, task_type="news_reasoning")
            
            return {
                "base_sentiment": base_sentiment,
                "llm_analysis": llm_analysis,
                "news_count": len(news),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {
                "base_sentiment": "NEUTRAL",
                "llm_analysis": "Analysis failed",
                "news_count": 0,
                "error": str(e)
            }

    def intelligent_decision_making(self, chart_signals: Dict, sentiment_data: Dict, 
                                 news: List[str], symbol: str) -> Dict[str, Any]:
        """Advanced LLM-based decision making"""
        
        # Construct comprehensive prompt for LLM
        decision_prompt = f"""
        TRADING DECISION ANALYSIS FOR {symbol}
        
        TECHNICAL ANALYSIS:
        - Short-term: {chart_signals.get('short', 'N/A')}
        - Medium-term: {chart_signals.get('medium', 'N/A')}  
        - Long-term: {chart_signals.get('long', 'N/A')}
        
        SENTIMENT ANALYSIS:
        - FinBERT Sentiment: {sentiment_data.get('base_sentiment', 'N/A')}
        - News Count: {sentiment_data.get('news_count', 0)}
        
        MARKET CONTEXT:
        - Recent News: {' | '.join(news[:3])}
        
        Provide a structured decision with:
        1. Action: BUY/SELL/HOLD/STRONG_BUY/STRONG_SELL
        2. Confidence: 0.0-1.0
        3. Reasoning: Brief explanation
        4. Risk Level: LOW/MEDIUM/HIGH
        5. Position Size: Percentage of portfolio
        
        Format: ACTION|CONFIDENCE|REASONING|RISK|SIZE
        """
        
        try:
            llm_response = route_llm_task(decision_prompt, task_type="strategy_decision")
            return self.parse_llm_decision(llm_response)
        except Exception as e:
            self.logger.error(f"LLM decision making failed: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": "Error in analysis",
                "risk_level": "HIGH",
                "position_size": 0.0
            }

    def parse_llm_decision(self, llm_response: str) -> Dict[str, Any]:
        """Parse structured LLM response"""
        try:
            # Try to parse structured format first
            if "|" in llm_response:
                parts = llm_response.split("|")
                if len(parts) >= 5:
                    return {
                        "action": parts[0].strip(),
                        "confidence": float(parts[1].strip()),
                        "reasoning": parts[2].strip(),
                        "risk_level": parts[3].strip(),
                        "position_size": float(parts[4].strip())
                    }
            
            # Fallback: extract key information from unstructured response
            action = "HOLD"
            confidence = 0.5
            
            response_lower = llm_response.lower()
            if "strong buy" in response_lower or "strong_buy" in response_lower:
                action = "STRONG_BUY"
                confidence = 0.8
            elif "buy" in response_lower:
                action = "BUY" 
                confidence = 0.7
            elif "strong sell" in response_lower or "strong_sell" in response_lower:
                action = "STRONG_SELL"
                confidence = 0.8
            elif "sell" in response_lower:
                action = "SELL"
                confidence = 0.7
                
            return {
                "action": action,
                "confidence": confidence,
                "reasoning": llm_response[:200],  # Truncate
                "risk_level": "MEDIUM",
                "position_size": 0.1
            }
            
        except Exception:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": "Failed to parse LLM response",
                "risk_level": "HIGH", 
                "position_size": 0.0
            }

    def comprehensive_risk_assessment(self, decision_data: Dict, symbol: str) -> Dict[str, Any]:
        """Enhanced risk management with multiple factors"""
        
        risk_prompt = f"""
        RISK ASSESSMENT FOR {symbol}
        
        Proposed Action: {decision_data.get('action')}
        Confidence: {decision_data.get('confidence')}
        Reasoning: {decision_data.get('reasoning')}
        
        Evaluate risks:
        1. Market Risk (volatility, correlations)
        2. Position Size Risk (portfolio impact)
        3. Timing Risk (entry/exit timing)
        4. Fundamental Risk (company-specific)
        5. Technical Risk (chart patterns)
        
        Provide:
        - Overall Risk Score: 0.0-1.0
        - Risk Factors: List key concerns
        - Risk Mitigation: Suggested safeguards
        - Final Recommendation: APPROVE/REDUCE/REJECT
        """
        
        try:
            risk_analysis = route_llm_task(risk_prompt, task_type="risk_management")
            
            # Apply position sizing rules
            base_position = decision_data.get('position_size', 0.1)
            risk_multiplier = 1.0
            
            if decision_data.get('confidence', 0) < 0.5:
                risk_multiplier *= 0.5  # Reduce for low confidence
                
            if decision_data.get('risk_level') == "HIGH":
                risk_multiplier *= 0.3  # Significantly reduce for high risk
                
            adjusted_position = base_position * risk_multiplier
            
            return {
                "original_decision": decision_data.get('action'),
                "risk_analysis": risk_analysis,
                "position_adjustment": risk_multiplier,
                "final_position_size": adjusted_position,
                "approval_status": "APPROVED" if adjusted_position > 0.01 else "REJECTED"
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return {
                "original_decision": decision_data.get('action'),
                "risk_analysis": "Risk assessment failed",
                "position_adjustment": 0.0,
                "final_position_size": 0.0,
                "approval_status": "REJECTED"
            }

    def run_comprehensive_analysis(self, symbol: str = "AAPL") -> TradingSignal:
        """Main orchestration method with comprehensive analysis"""
        
        self.logger.info(f"🚀 Starting comprehensive analysis for {symbol}")
        start_time = time.time()
        
        try:
            # Step 1: Multi-timeframe chart analysis
            self.logger.info("📊 Analyzing chart data across timeframes...")
            chart_signals = self.analyze_multiple_timeframes(symbol)
            
            # Step 2: Fetch and analyze news
            self.logger.info("📰 Fetching latest news...")
            news = fetch_news()  # Could be enhanced to fetch symbol-specific news
            
            # Step 3: Enhanced sentiment analysis
            self.logger.info("🧠 Running enhanced sentiment analysis...")
            sentiment_data = self.enhanced_sentiment_analysis(news, symbol)
            
            # Step 4: Intelligent decision making
            self.logger.info("🤖 Generating intelligent trading decision...")
            decision_data = self.intelligent_decision_making(
                chart_signals, sentiment_data, news, symbol
            )
            
            # Step 5: Comprehensive risk assessment
            self.logger.info("⚖️ Conducting comprehensive risk assessment...")
            risk_assessment = self.comprehensive_risk_assessment(decision_data, symbol)
            
            # Step 6: Create trading signal
            trading_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                chart_signal=str(chart_signals),
                sentiment=sentiment_data.get('base_sentiment', 'NEUTRAL'),
                news_summary=news[:5],  # Keep top 5 news items
                llm_reasoning=decision_data.get('reasoning', ''),
                risk_assessment=str(risk_assessment),
                final_decision=DecisionType(decision_data.get('action', 'HOLD')),
                confidence_score=decision_data.get('confidence', 0.0),
                metadata={
                    'processing_time': time.time() - start_time,
                    'chart_signals': chart_signals,
                    'sentiment_data': sentiment_data,
                    'decision_data': decision_data,
                    'risk_data': risk_assessment
                }
            )
            
            # Step 7: Log and store results
            self.trading_history.append(trading_signal)
            self.logger.info(f"✅ Analysis complete for {symbol}")
            self.logger.info(f"🎯 Final Decision: {trading_signal.final_decision.value}")
            self.logger.info(f"🔢 Confidence: {trading_signal.confidence_score:.2f}")
            
            # Step 8: Update strategy based on results
            update_strategy(trading_signal.final_decision.value)
            
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in analysis pipeline: {e}")
            # Return safe default signal
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                chart_signal="ERROR",
                sentiment="NEUTRAL", 
                news_summary=[],
                llm_reasoning=f"Analysis failed: {str(e)}",
                risk_assessment="HIGH_RISK",
                final_decision=DecisionType.HOLD,
                confidence_score=0.0,
                metadata={"error": str(e)}
            )

    def run_multi_symbol_analysis(self, symbols: Optional[List[str]] = None) -> List[TradingSignal]:
        """Run analysis on multiple symbols"""
        if symbols is None:
            symbols = self.settings.get('symbols', ['AAPL'])
            
        results = []
        for symbol in symbols:
            self.logger.info(f"🔄 Processing {symbol}...")
            signal = self.run_comprehensive_analysis(symbol)
            results.append(signal)
            time.sleep(1)  # Rate limiting
            
        return results

    def export_results(self, signals: List[TradingSignal], filepath: str = "data/analysis_results.json"):
        """Export analysis results to file"""
        try:
            results_data = []
            for signal in signals:
                results_data.append({
                    'timestamp': signal.timestamp.isoformat(),
                    'symbol': signal.symbol,
                    'final_decision': signal.final_decision.value,
                    'confidence_score': signal.confidence_score,
                    'reasoning': signal.llm_reasoning,
                    'metadata': signal.metadata
                })
                
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
                
            self.logger.info(f"📁 Results exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")

# Enhanced run function
def run():
    """Enhanced main execution function"""
    orchestrator = EnhancedOrchestrator()
    
    # Run analysis on configured symbols
    symbols = orchestrator.settings.get('symbols', ['AAPL'])
    results = orchestrator.run_multi_symbol_analysis(symbols)
    
    # Export results
    orchestrator.export_results(results)
    
    # Summary
    print("\n" + "="*50)
    print("📊 TRADING ANALYSIS SUMMARY")
    print("="*50)
    
    for result in results:
        print(f"Symbol: {result.symbol}")
        print(f"Decision: {result.final_decision.value}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Processing Time: {result.metadata.get('processing_time', 0):.2f}s")
        print("-" * 30)

if __name__ == "__main__":
    run()