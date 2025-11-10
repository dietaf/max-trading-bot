# ===================================================================
# BOT DE TRADING AUTOMATIZADO - MAX WAY STRATEGIES
# Version: 2.0 - Full Automation + Replit Compatible
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
import os
from collections import deque

# Importar Alpaca (se instalará automáticamente)
try:
    import alpaca_trade_api as tradeapi
except:
    st.error("Instalando Alpaca Trade API...")
    os.system("pip install alpaca-trade-api")
    import alpaca_trade_api as tradeapi

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

st.set_page_config(
    page_title="Max Way Bot - LIVE",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# CLASE DEL BOT AUTOMATIZADO
# ===================================================================

class AutomatedTradingBot:
    def __init__(self, api_key, api_secret, paper=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.api = None
        self.is_running = False
        self.current_position = None
        self.trades_history = []
        self.equity_history = deque(maxlen=1000)
        self.logs = deque(maxlen=100)
        
        # Configuración
        self.symbol = "SPY"
        self.timeframe = "15Min"
        self.strategy = "ORB"
        self.capital = 10000
        self.risk_per_trade = 0.02  # 2%
        self.trailing_stop_mult = 2.0
        self.take_profit_mult = 3.0
        
        # Estado del bot
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.position_size = 0
        
        if api_key and api_secret:
            base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
            try:
                self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
                self.log("✅ Conectado a Alpaca", "success")
            except Exception as e:
                self.log(f"❌ Error de conexión: {str(e)}", "error")
    
    def log(self, message, level="info"):
        """Registra eventos con timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            'time': timestamp,
            'message': message,
            'level': level
        })
    
    def get_account_info(self):
        """Obtiene información de la cuenta"""
        try:
            account = self.api.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity)
            }
        except Exception as e:
            self.log(f"Error obteniendo cuenta: {str(e)}", "error")
            return None
    
    def get_current_price(self, symbol):
        """Obtiene el precio actual"""
        try:
            quote = self.api.get_latest_trade(symbol)
            return quote.price
        except Exception as e:
            self.log(f"Error obteniendo precio: {str(e)}", "error")
            return None
    
    def get_historical_data(self, symbol, timeframe='15Min', limit=100):
        """Obtiene datos históricos"""
        try:
            end = datetime.now()
            start = end - timedelta(days=7)
            
            barset = self.api.get_bars(
                symbol,
                timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=limit
            ).df
            
            return barset
        except Exception as e:
            self.log(f"Error obteniendo datos: {str(e)}", "error")
            return None
    
    def calculate_atr(self, df, period=14):
        """Calcula ATR para stop loss dinámico"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if len(atr) > 0 else 0
    
    # ===============================================================
    # ESTRATEGIAS DE TRADING
    # ===============================================================
    
    def check_orb_signal(self, df):
        """Estrategia ORB - Opening Range Breakout"""
        if len(df) < 2:
            return 0
        
        # High y Low de la primera barra (15 min)
        orb_high = df['high'].iloc[0]
        orb_low = df['low'].iloc[0]
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        # Señal de compra: rompe el high
        if prev_price <= orb_high and current_price > orb_high:
            return 1
        
        # Señal de venta: rompe el low
        elif prev_price >= orb_low and current_price < orb_low:
            return -1
        
        return 0
    
    def check_pivot_signal(self, df, window=20):
        """Estrategia Pivot Hunter"""
        if len(df) < window * 2:
            return 0
        
        # Encontrar último pivot high y low
        pivot_high = df['high'].iloc[-window*2:-window].max()
        pivot_low = df['low'].iloc[-window*2:-window].min()
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        # Ruptura de resistencia
        if prev_price <= pivot_high and current_price > pivot_high:
            return 1
        
        # Ruptura de soporte
        elif prev_price >= pivot_low and current_price < pivot_low:
            return -1
        
        return 0
    
    def check_gap_signal(self, df, threshold=0.5):
        """Estrategia Gap Trading"""
        if len(df) < 2:
            return 0
        
        prev_close = df['close'].iloc[-2]
        current_open = df['open'].iloc[-1]
        
        gap_pct = ((current_open - prev_close) / prev_close) * 100
        
        if gap_pct > threshold:
            return 1
        elif gap_pct < -threshold:
            return -1
        
        return 0
    
    def check_quantum_signal(self, df, rsi_period=14):
        """Estrategia Quantum Shift"""
        if len(df) < rsi_period + 20:
            return 0
        
        # Calcular RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Volumen promedio
        vol_ma = df['volume'].rolling(window=20).mean()
        vol_ratio = df['volume'].iloc[-1] / vol_ma.iloc[-1]
        
        current_rsi = rsi.iloc[-1]
        
        # Sobreventa + volumen alto
        if current_rsi < 30 and vol_ratio > 1.5:
            return 1
        
        # Sobrecompra + volumen alto
        elif current_rsi > 70 and vol_ratio > 1.5:
            return -1
        
        return 0
    
    def check_trendshift_signal(self, df, fast=9, slow=21):
        """Estrategia TrendShift"""
        if len(df) < slow + 5:
            return 0
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # Golden cross
        if ema_fast.iloc[-2] <= ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            return 1
        
        # Death cross
        elif ema_fast.iloc[-2] >= ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            return -1
        
        return 0
    
    def get_signal(self, df):
        """Obtiene señal según la estrategia seleccionada"""
        if self.strategy == "ORB":
            return self.check_orb_signal(df)
        elif self.strategy == "Pivot Hunter":
            return self.check_pivot_signal(df)
        elif self.strategy == "Gap Trading":
            return self.check_gap_signal(df)
        elif self.strategy == "Quantum Shift":
            return self.check_quantum_signal(df)
        elif self.strategy == "TrendShift":
            return self.check_trendshift_signal(df)
        return 0
    
    # ===============================================================
    # EJECUCIÓN DE TRADES
    # ===============================================================
    
    def calculate_position_size(self, price, atr):
        """Calcula el tamaño de posición basado en riesgo"""
        account = self.get_account_info()
        if not account:
            return 0
        
        risk_amount = account['equity'] * self.risk_per_trade
        stop_distance = atr * self.trailing_stop_mult
        
        if stop_distance == 0:
            return 0
        
        shares = int(risk_amount / stop_distance)
        
        # Máximo 10% del capital en una posición
        max_shares = int(account['equity'] * 0.1 / price)
        
        return min(shares, max_shares)
    
    def place_order(self, symbol, qty, side):
        """Coloca una orden en Alpaca"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            self.log(f"✅ Orden ejecutada: {side} {qty} {symbol}", "success")
            return order
        except Exception as e:
            self.log(f"❌ Error ejecutando orden: {str(e)}", "error")
            return None
    
    def open_position(self, signal, price, atr):
        """Abre una nueva posición"""
        qty = self.calculate_position_size(price, atr)
        
        if qty == 0:
            self.log("⚠️ Posición muy pequeña, skip", "warning")
            return
        
        side = "buy" if signal == 1 else "sell"
        order = self.place_order(self.symbol, qty, side)
        
        if order:
            self.current_position = {
                'type': 'LONG' if signal == 1 else 'SHORT',
                'entry_price': price,
                'qty': qty,
                'entry_time': datetime.now()
            }
            
            # Calcular stops
            if signal == 1:  # LONG
                self.stop_loss = price - (atr * self.trailing_stop_mult)
                self.take_profit = price + (atr * self.take_profit_mult)
            else:  # SHORT
                self.stop_loss = price + (atr * self.trailing_stop_mult)
                self.take_profit = price - (atr * self.take_profit_mult)
            
            self.log(f"🎯 Posición abierta: {self.current_position['type']} @ ${price:.2f}", "success")
            self.log(f"   Stop: ${self.stop_loss:.2f} | TP: ${self.take_profit:.2f}", "info")
    
    def update_trailing_stop(self, current_price, atr):
        """Actualiza el trailing stop"""
        if not self.current_position:
            return
        
        if self.current_position['type'] == 'LONG':
            new_stop = current_price - (atr * self.trailing_stop_mult)
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
                self.log(f"📈 Trailing stop actualizado: ${self.stop_loss:.2f}", "info")
        
        else:  # SHORT
            new_stop = current_price + (atr * self.trailing_stop_mult)
            if new_stop < self.stop_loss:
                self.stop_loss = new_stop
                self.log(f"📉 Trailing stop actualizado: ${self.stop_loss:.2f}", "info")
    
    def close_position(self, reason, current_price):
        """Cierra la posición actual"""
        if not self.current_position:
            return
        
        side = "sell" if self.current_position['type'] == 'LONG' else "buy"
        qty = self.current_position['qty']
        
        order = self.place_order(self.symbol, qty, side)
        
        if order:
            # Calcular profit
            if self.current_position['type'] == 'LONG':
                profit = current_price - self.current_position['entry_price']
            else:
                profit = self.current_position['entry_price'] - current_price
            
            profit_pct = (profit / self.current_position['entry_price']) * 100
            profit_usd = profit * qty
            
            # Guardar trade
            self.trades_history.append({
                'entry_time': self.current_position['entry_time'],
                'exit_time': datetime.now(),
                'type': self.current_position['type'],
                'entry_price': self.current_position['entry_price'],
                'exit_price': current_price,
                'qty': qty,
                'profit': profit_usd,
                'profit_pct': profit_pct,
                'reason': reason
            })
            
            emoji = "💰" if profit_usd > 0 else "💸"
            self.log(f"{emoji} Posición cerrada: {reason}", "success" if profit_usd > 0 else "error")
            self.log(f"   P/L: ${profit_usd:.2f} ({profit_pct:+.2f}%)", "info")
            
            self.current_position = None
            self.stop_loss = 0
            self.take_profit = 0
    
    def check_exit_conditions(self, current_price):
        """Verifica condiciones de salida"""
        if not self.current_position:
            return
        
        if self.current_position['type'] == 'LONG':
            # Stop loss
            if current_price <= self.stop_loss:
                self.close_position("Stop Loss", current_price)
            # Take profit
            elif current_price >= self.take_profit:
                self.close_position("Take Profit", current_price)
        
        else:  # SHORT
            # Stop loss
            if current_price >= self.stop_loss:
                self.close_position("Stop Loss", current_price)
            # Take profit
            elif current_price <= self.take_profit:
                self.close_position("Take Profit", current_price)
    
    # ===============================================================
    # LOOP PRINCIPAL DEL BOT
    # ===============================================================
    
    def trading_loop(self):
        """Loop principal que monitorea y ejecuta trades"""
        self.log("🤖 Bot iniciado en modo automático", "success")
        
        while self.is_running:
            try:
                # Verificar horario de mercado
                clock = self.api.get_clock()
                if not clock.is_open:
                    self.log("⏰ Mercado cerrado, esperando...", "warning")
                    time.sleep(60)
                    continue
                
                # Obtener datos actuales
                df = self.get_historical_data(self.symbol, self.timeframe)
                if df is None or len(df) == 0:
                    time.sleep(30)
                    continue
                
                current_price = self.get_current_price(self.symbol)
                if not current_price:
                    time.sleep(30)
                    continue
                
                atr = self.calculate_atr(df)
                
                # Actualizar equity
                account = self.get_account_info()
                if account:
                    self.equity_history.append(account['equity'])
                
                # Si tenemos posición, verificar salida
                if self.current_position:
                    self.check_exit_conditions(current_price)
                    self.update_trailing_stop(current_price, atr)
                
                # Si no tenemos posición, buscar entrada
                else:
                    signal = self.get_signal(df)
                    if signal != 0:
                        self.log(f"🎯 Señal detectada: {'COMPRA' if signal == 1 else 'VENTA'}", "info")
                        self.open_position(signal, current_price, atr)
                
                # Esperar antes del próximo ciclo
                time.sleep(30)  # Revisa cada 30 segundos
                
            except Exception as e:
                self.log(f"❌ Error en loop: {str(e)}", "error")
                time.sleep(60)
        
        self.log("🛑 Bot detenido", "warning")
    
    def start(self):
        """Inicia el bot en un thread separado"""
        if not self.is_running:
            self.is_running = True
            thread = threading.Thread(target=self.trading_loop, daemon=True)
            thread.start()
            return True
        return False
    
    def stop(self):
        """Detiene el bot"""
        self.is_running = False
        # Cerrar posición si existe
        if self.current_position:
            price = self.get_current_price(self.symbol)
            if price:
                self.close_position("Bot detenido", price)

# ===================================================================
# INTERFAZ STREAMLIT
# ===================================================================

def main():
    # Título
    st.title("🤖 Max Way Bot - LIVE TRADING")
    st.markdown("### Bot Automatizado con Estrategias de Big Daddy Max")
    
    # Inicializar bot en session state
    if 'bot' not in st.session_state:
        st.session_state.bot = None
        st.session_state.bot_running = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # API Keys
        st.subheader("🔑 Alpaca API")
        api_key = st.text_input("API Key", type="password", key="api_key")
        api_secret = st.text_input("API Secret", type="password", key="api_secret")
        paper_trading = st.checkbox("Paper Trading", value=True)
        
        st.divider()
        
        # Configuración del bot
        st.subheader("🎯 Estrategia")
        strategy = st.selectbox(
            "Estrategia",
            ["ORB", "Pivot Hunter", "Gap Trading", "Quantum Shift", "TrendShift"]
        )
        
        symbol = st.text_input("Símbolo", value="SPY")
        timeframe = st.selectbox("Timeframe", ["1Min", "5Min", "15Min", "1Hour"])
        
        st.divider()
        
        # Risk Management
        st.subheader("💰 Risk Management")
        risk_pct = st.slider("Riesgo por Trade (%)", 1, 5, 2) / 100
        trailing_mult = st.slider("Trailing Stop (ATR)", 1.0, 4.0, 2.0, 0.5)
        tp_mult = st.slider("Take Profit (ATR)", 1.0, 5.0, 3.0, 0.5)
        
        st.divider()
        
        # Controles
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ INICIAR", use_container_width=True, type="primary"):
                if api_key and api_secret:
                    st.session_state.bot = AutomatedTradingBot(api_key, api_secret, paper_trading)
                    st.session_state.bot.strategy = strategy
                    st.session_state.bot.symbol = symbol
                    st.session_state.bot.timeframe = timeframe
                    st.session_state.bot.risk_per_trade = risk_pct
                    st.session_state.bot.trailing_stop_mult = trailing_mult
                    st.session_state.bot.take_profit_mult = tp_mult
                    
                    if st.session_state.bot.start():
                        st.session_state.bot_running = True
                        st.success("✅ Bot iniciado!")
                else:
                    st.error("⚠️ Ingresa tus API keys")
        
        with col2:
            if st.button("⏹️ DETENER", use_container_width=True):
                if st.session_state.bot:
                    st.session_state.bot.stop()
                    st.session_state.bot_running = False
                    st.warning("🛑 Bot detenido")
    
    # Main area
    if st.session_state.bot and st.session_state.bot_running:
        bot = st.session_state.bot
        
        # Status indicator
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #00ff00 0%, #00cc00 100%); 
                    padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">🟢 BOT ACTIVO - {bot.strategy} en {bot.symbol}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas
        account = bot.get_account_info()
        if account:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💵 Balance", f"${account['equity']:,.2f}")
            with col2:
                position_status = "LONG" if bot.current_position and bot.current_position['type'] == 'LONG' else "SHORT" if bot.current_position else "Sin posición"
                st.metric("📊 Posición", position_status)
            with col3:
                total_trades = len(bot.trades_history)
                st.metric("📈 Trades", total_trades)
            with col4:
                if bot.trades_history:
                    winning = len([t for t in bot.trades_history if t['profit'] > 0])
                    win_rate = (winning / total_trades * 100)
                    st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
                else:
                    st.metric("🎯 Win Rate", "0%")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📜 Logs", "💼 Trades"])
        
        with tab1:
            # Equity curve
            if len(bot.equity_history) > 1:
                st.subheader("📈 Curva de Capital")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=list(bot.equity_history),
                    mode='lines',
                    name='Equity',
                    line=dict(color='#00ff00', width=2),
                    fill='tozeroy'
                ))
                fig.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            # Posición actual
            if bot.current_position:
                st.subheader("📍 Posición Actual")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"""
                    **Tipo:** {bot.current_position['type']}  
                    **Entrada:** ${bot.current_position['entry_price']:.2f}  
                    **Cantidad:** {bot.current_position['qty']} shares
                    """)
                
                with col2:
                    current_price = bot.get_current_price(bot.symbol)
                    if current_price:
                        if bot.current_position['type'] == 'LONG':
                            pnl = (current_price - bot.current_position['entry_price']) * bot.current_position['qty']
                        else:
                            pnl = (bot.current_position['entry_price'] - current_price) * bot.current_position['qty']
                        
                        color = "green" if pnl > 0 else "red"
                        st.markdown(f"""
                        **Precio Actual:** ${current_price:.2f}  
                        **P/L:** <span style="color: {color};">${pnl:.2f}</span>
                        """, unsafe_allow_html=True)
                
                with col3:
                    st.warning(f"""
                    **Stop Loss:** ${bot.stop_loss:.2f}  
                    **Take Profit:** ${bot.take_profit:.2f}
                    """)
        
        with tab2:
            st.subheader("📜 Logs en Tiempo Real")
            
            # Mostrar logs
            logs_container = st.container()
            with logs_container:
                for log in reversed(list(bot.logs)):
                    color = {
                        'success': 'green',
                        'error': 'red',
                        'warning': 'orange',
                        'info': 'blue'
                    }.get(log['level'], 'white')
                    
                    st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.3); padding: 8px; margin: 4px 0; border-radius: 5px; border-left: 3px solid {color};">
                        <span style="color: gray;">[{log['time']}]</span> {log['message']}
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.subheader("💼 Historial de Trades")
            
            if bot.trades_history:
                df_trades = pd.DataFrame(bot.trades_history)
                
                # Calcular estadísticas
                total_profit = df_trades['profit'].sum()
                avg_profit = df_trades['profit'].mean()
                winning_trades = len(df_trades[df_trades['profit'] > 0])
                losing_trades = len(df_trades[df_trades['profit'] < 0])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total P/L", f"${total_profit:.2f}")
                with col2:
                    st.metric("Avg P/L", f"${avg_profit:.2f}")
                with col3:
                    st.metric("✅ Wins", winning_trades)
                with col4:
                    st.metric("❌ Losses", losing_trades)
                
                st.dataframe(df_trades, use_container_width=True)
            else:
                st.info("📊 No hay trades ejecutados aún")
        
        # Auto-refresh cada 5 segundos
        time.sleep(5)
        st.rerun()
    
    else:
        # Mensaje inicial
        st.info("""
        ### 🚀 Bienvenido al Bot Automatizado
        
        **Para comenzar:**
        1. Ingresa tus API keys de Alpaca en el sidebar
        2. Configura tu estrategia y parámetros
        3. Click en ▶️ INICIAR
        
        El bot monitoreará el mercado 24/7 y ejecutará trades automáticamente.
        """)

if __name__ == "__main__":
    main()
