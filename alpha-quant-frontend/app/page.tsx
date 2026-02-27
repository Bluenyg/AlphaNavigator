"use client";

import React, { useState, useEffect } from "react";
import ReactECharts from "echarts-for-react";
import {
  Bot, PieChart, TrendingUp, Sparkles, Send,
  Wallet, ShieldCheck, Zap, Activity, User, LogOut, ArrowRight, Lock, AlertCircle, History, Calendar
} from "lucide-react";

const API_BASE = "http://localhost:8000/api";

const INDEX_CONFIG = [
  { name: "大盘/沪深300", code: "sh000300" },
  { name: "中盘/中证500", code: "sh000905" },
  { name: "科技/创业板", code: "sz399006" }
];

export default function AlphaQuantDashboard() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState("");
  const [loginInput, setLoginInput] = useState("");
  const [passwordInput, setPasswordInput] = useState("");
  const [authError, setAuthError] = useState("");
  const [isAuthenticating, setIsAuthenticating] = useState(false);

  const [portfolio, setPortfolio] = useState<any[]>([]);
  const [totalCost, setTotalCost] = useState(0);
  const [transactions, setTransactions] = useState<any[]>([]);

  const [selectedIndex, setSelectedIndex] = useState("sh000300");
  const [chartData, setChartData] = useState<any>(null);

  const [advisorData, setAdvisorData] = useState<any>(null);
  const [adviceHistory, setAdviceHistory] = useState<any[]>([]);
  const [selectedAdviceId, setSelectedAdviceId] = useState<number | string>("");
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [isAdvising, setIsAdvising] = useState(false);

  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState<{role: string, content: string}[]>([]);

  // 基金份额类别智能识别与标签渲染
  const renderFundClassBadge = (name: string) => {
    if (!name) return null;
    const upperName = name.toUpperCase();

    if (upperName.includes("C")) {
      return (
        <span
          className="inline-flex items-center ml-2 px-1.5 py-0.5 bg-amber-100 text-amber-600 text-[10px] font-black rounded border border-amber-200 tracking-wider whitespace-nowrap cursor-help"
          title="C类份额：0申购费，按日计提服务费。适合1-6个月波段 (⚠️提示：务必持有满7天以避开1.5%巨额惩罚赎回费)"
        >
          C类(波段)
        </span>
      );
    } else if (upperName.includes("A")) {
      return (
        <span
          className="inline-flex items-center ml-2 px-1.5 py-0.5 bg-indigo-100 text-indigo-600 text-[10px] font-black rounded border border-indigo-200 tracking-wider whitespace-nowrap cursor-help"
          title="A类份额：前端一次性收费。适合1年以上长线持有 (⚠️提示：短期赎回费极高，严禁频繁交易)"
        >
          A类(长线)
        </span>
      );
    }
    return null;
  };

  useEffect(() => {
    if (isLoggedIn && username) {
      if (chatHistory.length === 0) {
        setChatHistory([
          { role: "ai", content: `喵~ 欢迎回来，主理人 ${username}！告诉我你想买卖什么基金吧，例如：'买入10000元沪深300'` }
        ]);
      }

      fetchPortfolio();
      fetchAdvisorHistory(true);

      const timer = setInterval(() => {
        fetchPortfolio();
        fetchAdvisorHistory(false);
      }, 30000);

      return () => clearInterval(timer);
    }
  }, [isLoggedIn, username]);

  useEffect(() => {
    if (isLoggedIn && username) {
      fetchMarketChart(selectedIndex);
    }
  }, [isLoggedIn, username, selectedIndex]);

  const fetchPortfolio = async () => {
    try {
      const res = await fetch(`${API_BASE}/user/portfolio?username=${username}&_t=${Date.now()}`, {
        cache: 'no-store',
        headers: { 'Cache-Control': 'no-cache' }
      });
      const data = await res.json();
      if (data.status === "success") {
        setPortfolio(data.portfolio);
        setTotalCost(data.total_cost_basis || data.total_assets || 0);
        setTransactions(data.transactions || []);
      }
    } catch (error) {
      console.error("获取持仓失败", error);
    }
  };

  const fetchMarketChart = async (code: string) => {
    try {
      setChartData(null);
      const res = await fetch(`${API_BASE}/market/visualization?index_code=${code}&_t=${Date.now()}`, {
        cache: 'no-store',
        headers: { 'Cache-Control': 'no-cache' }
      });
      const data = await res.json();
      if (data.status === "success") setChartData(data);
    } catch (error) {
      console.error("获取图表失败", error);
    }
  };

  const fetchAdvisorHistory = async (forceSelectLatest = false) => {
    try {
      const res = await fetch(`${API_BASE}/advisor/history?username=${username}&_t=${Date.now()}`, {
        cache: 'no-store',
        headers: { 'Cache-Control': 'no-cache' }
      });
      const data = await res.json();
      if (data.status === "success") {
        setAdviceHistory(data.history);
        if (data.history.length > 0 && forceSelectLatest) {
          const latest = data.history[0];
          setAdvisorData(latest);
          setSelectedAdviceId(latest.id);
          setSelectedDate(latest.date.split(" ")[0]);
        }
      }
    } catch (error) {
      console.error("获取历史策略失败", error);
    }
  };

  const handleDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newDate = e.target.value;
    setSelectedDate(newDate);
    const matchedReports = adviceHistory.filter(h => h.date.startsWith(newDate));

    if (matchedReports.length > 0) {
      setAdvisorData(matchedReports[0]);
      setSelectedAdviceId(matchedReports[0].id);
    } else {
      setAdvisorData(null);
      setSelectedAdviceId("");
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    const newHistory = [...chatHistory, { role: "user", content: chatInput }];
    setChatHistory(newHistory);
    setChatInput("");
    try {
      const res = await fetch(`${API_BASE}/user/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, message: chatInput })
      });
      const data = await res.json();
      setChatHistory([...newHistory, { role: "ai", content: data.reply || "处理出错啦喵~" }]);
      fetchPortfolio();
    } catch (error) {
      setChatHistory([...newHistory, { role: "ai", content: "网络连接失败了..." }]);
    }
  };

  const handleGenerateAdvice = async () => {
    setIsAdvising(true);
    try {
      const res = await fetch(`${API_BASE}/advisor/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, force_refresh: false })
      });
      const data = await res.json();
      if (data.status === "success") {
        setAdvisorData(data);
        await fetchAdvisorHistory(true);
      }
    } catch (error) {
      console.error("生成建议失败", error);
    }
    setIsAdvising(false);
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!loginInput.trim() || !passwordInput.trim()) return;

    setIsAuthenticating(true);
    setAuthError("");

    try {
      const res = await fetch(`${API_BASE}/user/auth`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: loginInput.trim(),
          password: passwordInput
        })
      });

      const data = await res.json();

      if (res.ok && data.status === "success") {
        setUsername(loginInput.trim());
        setIsLoggedIn(true);
      } else {
        setAuthError(data.detail || "登录失败，请检查网络");
      }
    } catch (error) {
      setAuthError("无法连接到后端服务器");
    } finally {
      setIsAuthenticating(false);
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUsername("");
    setLoginInput("");
    setPasswordInput("");
    setPortfolio([]);
    setTransactions([]);
    setAdvisorData(null);
    setAdviceHistory([]);
    setSelectedDate("");
    setAuthError("");
    setChatHistory([]);
  };

  // 🌟 核心重构：专业量化主副图（双 Grid）配置
  const getChartOptions = () => {
    if (!chartData || !chartData.xAxis_dates || chartData.xAxis_dates.length === 0) return {};

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderRadius: 8,
        textStyle: { color: '#334155' }
      },
      legend: {
        data: ['价格均线 (MA20)', 'RSI (超买超卖)', 'MACD (趋势动能)'],
        top: 0,
        textStyle: { fontSize: 12, color: '#64748b' }
      },
      // 🌟 切割为两个区域：主图占 50%，副图占 25%
      grid: [
        { left: '6%', right: '4%', top: '12%', height: '50%', containLabel: true },
        { left: '6%', right: '4%', top: '68%', height: '25%', containLabel: true }
      ],
      xAxis: [
        {
          type: 'category',
          data: chartData.xAxis_dates,
          gridIndex: 0,
          axisLabel: { show: false }, // 隐藏主图的时间轴，让副图显示即可
          axisTick: { show: false },
          axisLine: { lineStyle: { color: '#e2e8f0' } }
        },
        {
          type: 'category',
          data: chartData.xAxis_dates,
          gridIndex: 1, // 副图的时间轴
          axisLine: { lineStyle: { color: '#e2e8f0' } },
          axisLabel: { color: '#64748b', fontSize: 10 }
        }
      ],
      yAxis: [
        { // 主图 Y 轴 (MA20)
          type: 'value',
          gridIndex: 0,
          scale: true, // 🌟 绝对关键：开启自适应缩放，不再从 0 绘制，解决一条直线的 Bug
          splitLine: { lineStyle: { type: 'dashed', color: '#f1f5f9' } },
          axisLabel: { color: '#64748b', fontSize: 10 }
        },
        { // 副图 Y 轴左侧 (MACD)
          type: 'value',
          gridIndex: 1,
          scale: true,
          splitLine: { show: false },
          axisLabel: { color: '#64748b', fontSize: 10 }
        },
        { // 副图 Y 轴右侧 (RSI)
          type: 'value',
          gridIndex: 1,
          position: 'right',
          min: 0,
          max: 100, // RSI 严格限制在 0-100 区间
          splitLine: { show: false },
          axisLabel: { show: false }
        }
      ],
      series: [
        {
          name: '价格均线 (MA20)',
          type: 'line',
          data: chartData.series.ma20,
          xAxisIndex: 0,
          yAxisIndex: 0,
          smooth: true,
          itemStyle: { color: '#8b5cf6' },
          lineStyle: { width: 2.5 },
          symbol: 'none'
        },
        {
          name: 'MACD (趋势动能)',
          type: 'bar',
          data: chartData.series.macd,
          xAxisIndex: 1,
          yAxisIndex: 1, // 挂载到副图
          itemStyle: {
            // A股习惯：红涨绿跌
            color: (params: any) => params.value > 0 ? '#ef4444' : '#10b981',
            borderRadius: [2, 2, 0, 0]
          }
        },
        {
          name: 'RSI (超买超卖)',
          type: 'line',
          data: chartData.series.rsi,
          xAxisIndex: 1,
          yAxisIndex: 2, // 挂载到副图的右侧固定轴
          smooth: true,
          itemStyle: { color: '#f59e0b' },
          lineStyle: { width: 1.5, type: 'dashed' },
          symbol: 'none'
        }
      ]
    };
  };

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 flex items-center justify-center p-6 relative overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-purple-300/40 rounded-full blur-[100px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-pink-300/40 rounded-full blur-[100px]"></div>

        <div className="w-full max-w-md bg-white/60 backdrop-blur-2xl rounded-[2.5rem] p-10 shadow-2xl shadow-indigo-100/50 border border-white relative z-10 animate-fade-in">
          <div className="flex justify-center mb-6">
            <div className="bg-gradient-to-tr from-violet-500 to-fuchsia-500 p-4 rounded-3xl shadow-xl shadow-purple-200">
              <Sparkles className="text-white w-10 h-10" />
            </div>
          </div>
          <h1 className="text-3xl font-black text-center bg-clip-text text-transparent bg-gradient-to-r from-violet-600 to-fuchsia-600 mb-2">
            Alpha Quant
          </h1>
          <p className="text-center text-slate-500 text-sm font-medium mb-8">
            下一代多智能体量化投顾平台
          </p>

          {authError && (
            <div className="mb-6 p-3 bg-rose-50 border border-rose-200 rounded-xl flex items-center gap-2 text-rose-600 text-sm font-bold animate-fade-in">
              <AlertCircle className="w-4 h-4" />
              {authError}
            </div>
          )}

          <form onSubmit={handleLogin} className="space-y-5">
            <div>
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2 ml-2">主理人代号</label>
              <div className="relative">
                <User className="absolute left-4 top-3.5 w-5 h-5 text-slate-400" />
                <input
                  type="text"
                  value={loginInput}
                  onChange={(e) => setLoginInput(e.target.value)}
                  placeholder="请输入账户名"
                  required
                  className="w-full bg-white/80 border border-slate-200 text-slate-700 rounded-2xl py-3 pl-12 pr-4 focus:outline-none focus:ring-2 focus:ring-fuchsia-300 shadow-sm transition"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2 ml-2">安全密码</label>
              <div className="relative">
                <Lock className="absolute left-4 top-3.5 w-5 h-5 text-slate-400" />
                <input
                  type="password"
                  value={passwordInput}
                  onChange={(e) => setPasswordInput(e.target.value)}
                  placeholder="请输入您的安全密码"
                  required
                  className="w-full bg-white/80 border border-slate-200 text-slate-700 rounded-2xl py-3 pl-12 pr-4 focus:outline-none focus:ring-2 focus:ring-fuchsia-300 shadow-sm transition"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isAuthenticating}
              className="w-full mt-2 flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-900 text-white py-3.5 rounded-2xl font-bold shadow-lg shadow-slate-300 transition transform hover:-translate-y-0.5 disabled:opacity-70 disabled:transform-none"
            >
              {isAuthenticating ? "正在安全通讯..." : <><ArrowRight className="w-4 h-4" /> 进入系统</>}
            </button>
          </form>
          <div className="mt-8 text-center">
            <p className="text-xs text-slate-400">若账户不存在，系统将使用该密码自动注册</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 p-6 md:p-10 font-sans text-slate-800 animate-fade-in">
      <header className="flex items-center justify-between mb-10 bg-white/60 backdrop-blur-xl p-4 md:px-8 rounded-3xl shadow-sm border border-white">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-tr from-violet-500 to-fuchsia-500 p-2.5 rounded-2xl shadow-lg shadow-purple-200">
            <Sparkles className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-black bg-clip-text text-transparent bg-gradient-to-r from-violet-600 to-fuchsia-600">
              Alpha Quant
            </h1>
            <p className="text-xs text-slate-500 font-medium hidden md:block">量化驱动 · 多智能体投顾</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-indigo-50 px-4 py-2 rounded-full border border-indigo-100">
            <div className="w-2.5 h-2.5 bg-emerald-400 rounded-full animate-pulse"></div>
            <span className="text-sm font-bold text-indigo-900">{username}</span>
          </div>
          <button
            onClick={handleLogout}
            title="退出登录"
            className="p-2 bg-white rounded-full text-slate-400 hover:text-rose-500 hover:bg-rose-50 border border-slate-100 transition shadow-sm"
          >
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-8">
          <div className="bg-white/70 backdrop-blur-lg rounded-[2rem] p-6 shadow-xl shadow-indigo-100/50 border border-white">
            <div className="flex items-center gap-2 mb-6">
              <Wallet className="w-5 h-5 text-indigo-500" />
              <h2 className="text-lg font-bold text-slate-700">我的小金库</h2>
            </div>

            <div className="space-y-6">
              <div className="flex gap-4 mb-6">
                <div className="flex-1 bg-white p-4 rounded-2xl shadow-sm border border-slate-100">
                  <p className="text-xs text-slate-400 font-medium mb-1 uppercase">投入本金</p>
                  <p className="text-xl font-bold text-slate-600">¥{totalCost.toLocaleString()}</p>
                </div>
                <div className="flex-1 bg-gradient-to-br from-indigo-500 to-fuchsia-500 p-4 rounded-2xl shadow-md text-white">
                  <p className="text-xs text-indigo-100 font-medium mb-1 uppercase">当前总市值</p>
                  <p className="text-2xl font-black">
                    ¥{portfolio.reduce((sum, item) => sum + (item.current_value || item.cost_basis), 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                  </p>
                </div>
              </div>

              <div>
                <h3 className="text-xs font-bold text-slate-400 mb-3 uppercase tracking-wider">📊 当前持仓明细</h3>
                <div className="space-y-3">
                  {portfolio.length === 0 ? (
                    <div className="text-center py-4 text-slate-400 text-sm bg-slate-50/50 rounded-2xl border border-dashed border-slate-200">持仓为空，快让精灵买点什么吧~</div>
                  ) : (
                    portfolio.map((item, idx) => (
                      <div key={idx} className="flex justify-between items-center p-3 bg-white rounded-2xl shadow-sm border border-slate-50 hover:shadow-md transition">
                        <div>
                          <p className="text-sm font-bold text-slate-700 flex items-center">
                            {item.fund_name}
                            {renderFundClassBadge(item.fund_name)}
                          </p>
                          <p className="text-xs text-slate-400 mt-1">{item.fund_code} · {item.shares}份</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-bold text-indigo-600">¥{item.current_value || item.cost_basis}</p>
                          <p className={`text-xs font-bold ${item.profit_rate >= 0 ? 'text-rose-500' : 'text-emerald-500'}`}>
                            {item.profit_rate >= 0 ? '+' : ''}{item.profit_rate}%
                          </p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div>
                <h3 className="text-xs font-bold text-slate-400 mb-3 uppercase tracking-wider">⏳ 最近交易</h3>
                <div className="space-y-2 max-h-[150px] overflow-y-auto scrollbar-thin pr-1">
                  {transactions.length === 0 ? (
                    <div className="text-center py-2 text-slate-400 text-xs">暂无交易记录</div>
                  ) : (
                    transactions.map((tx, idx) => (
                      <div key={idx} className="flex justify-between items-center p-2.5 bg-slate-50/50 rounded-xl border border-slate-100">
                        <div className="flex items-center gap-2">
                          <span className={`text-xs font-black px-1.5 py-0.5 rounded ${tx.action === '买入' ? 'bg-rose-100 text-rose-600' : 'bg-emerald-100 text-emerald-600'}`}>
                            {tx.action}
                          </span>
                          <span className="text-xs font-medium text-slate-600">{tx.fund_code}</span>
                        </div>
                        <div className="text-right">
                          <p className="text-xs font-bold text-slate-700">¥{tx.amount}</p>
                          <p className="text-[10px] text-slate-400">{tx.time}</p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white/70 backdrop-blur-lg rounded-[2rem] p-6 shadow-xl shadow-pink-100/50 border border-white flex flex-col h-[400px]">
            <div className="flex items-center gap-2 mb-4">
              <Bot className="w-5 h-5 text-fuchsia-500" />
              <h2 className="text-lg font-bold text-slate-700">交易小精灵</h2>
            </div>
            <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2 scrollbar-thin">
              {chatHistory.map((chat, idx) => (
                <div key={idx} className={`flex ${chat.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] p-3 rounded-2xl text-sm ${
                    chat.role === 'user' 
                      ? 'bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white rounded-tr-sm shadow-md' 
                      : 'bg-white text-slate-700 rounded-tl-sm shadow-sm border border-slate-100'
                  }`}>
                    {chat.content}
                  </div>
                </div>
              ))}
            </div>
            <form onSubmit={handleChatSubmit} className="relative">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="发送交易指令..."
                className="w-full bg-white/80 border border-slate-200 text-sm rounded-full py-3 pl-4 pr-12 focus:outline-none focus:ring-2 focus:ring-fuchsia-300 shadow-inner"
              />
              <button type="submit" className="absolute right-2 top-1.5 p-1.5 bg-fuchsia-500 text-white rounded-full hover:bg-fuchsia-600 transition shadow-md">
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-8">

          <div className="bg-white/70 backdrop-blur-lg rounded-[2rem] p-6 shadow-xl shadow-blue-100/50 border border-white">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-500" />
                <h2 className="text-lg font-bold text-slate-700">市场量化雷达</h2>
              </div>

              <div className="flex bg-slate-100/80 p-1 rounded-xl shadow-inner border border-slate-200/50">
                {INDEX_CONFIG.map((idx) => (
                  <button
                    key={idx.code}
                    onClick={() => setSelectedIndex(idx.code)}
                    className={`px-4 py-1.5 text-xs font-bold rounded-lg transition-all duration-300 ${
                      selectedIndex === idx.code 
                        ? 'bg-white text-indigo-600 shadow-sm' 
                        : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
                    }`}
                  >
                    {idx.name}
                  </button>
                ))}
              </div>
            </div>

            {/* 🌟 增加 overflow-hidden 防止容器意外溢出 */}
            <div className="h-[350px] w-full overflow-hidden">
              {chartData ? (
                <ReactECharts option={getChartOptions()} style={{ height: '100%', width: '100%' }} />
              ) : (
                <div className="h-full flex items-center justify-center text-slate-400 animate-pulse">
                  正在切换频道，获取最新量价数据喵...
                </div>
              )}
            </div>
          </div>

          <div className="bg-gradient-to-br from-white to-indigo-50/50 backdrop-blur-lg rounded-[2rem] p-6 shadow-xl shadow-indigo-100/50 border border-white relative overflow-hidden">
            <div className="absolute -right-10 -top-10 w-40 h-40 bg-fuchsia-200/30 rounded-full blur-3xl"></div>

            <div className="flex justify-between items-center mb-6 relative z-10 flex-wrap gap-4">
              <div className="flex items-center gap-3">
                <ShieldCheck className="w-6 h-6 text-indigo-600" />
                <h2 className="text-xl font-bold text-slate-800">智能策略室</h2>

                {adviceHistory.length > 0 && (
                  <div className="flex items-center bg-white border border-indigo-100 hover:border-indigo-300 transition-colors rounded-full px-3 py-1.5 shadow-sm ml-2">
                    <Calendar className="w-4 h-4 text-indigo-500 mr-2" />
                    <input
                      type="date"
                      className="bg-transparent text-sm font-bold text-slate-700 outline-none cursor-pointer [&::-webkit-calendar-picker-indicator]:cursor-pointer [&::-webkit-calendar-picker-indicator]:opacity-60 hover:[&::-webkit-calendar-picker-indicator]:opacity-100 transition-opacity"
                      value={selectedDate}
                      onChange={handleDateChange}
                      title="按日期查找历史研报"
                    />

                    {adviceHistory.filter(h => h.date.startsWith(selectedDate)).length > 1 && (
                      <select
                        className="bg-transparent text-xs font-medium text-slate-500 outline-none cursor-pointer ml-2 border-l border-slate-200 pl-2"
                        value={selectedAdviceId}
                        onChange={(e) => {
                          const id = e.target.value;
                          setSelectedAdviceId(id);
                          const selected = adviceHistory.find(h => h.id.toString() === id);
                          if(selected) setAdvisorData(selected);
                        }}
                      >
                        {adviceHistory
                          .filter(h => h.date.startsWith(selectedDate))
                          .map(h => (
                            <option key={h.id} value={h.id}>{h.date.split(" ")[1]}</option>
                          ))}
                      </select>
                    )}
                  </div>
                )}
              </div>

              <button
                onClick={handleGenerateAdvice}
                disabled={isAdvising}
                className="flex items-center gap-2 bg-slate-800 hover:bg-slate-900 text-white px-5 py-2.5 rounded-full text-sm font-bold shadow-lg shadow-slate-300 transition transform hover:scale-105 disabled:opacity-50"
              >
                {isAdvising ? "大脑飞速运转中..." : <><Zap className="w-4 h-4 text-amber-400" /> 生成今日策略</>}
              </button>
            </div>

            {advisorData ? (
              <div className="space-y-6 relative z-10 animate-fade-in">
                <div className="bg-white/80 p-5 rounded-2xl border border-indigo-50 shadow-sm">
                  <h3 className="text-sm font-bold text-indigo-900 mb-2 flex items-center gap-2"><TrendingUp className="w-4 h-4"/> 宏观环境研判</h3>
                  <p className="text-sm text-slate-600 leading-relaxed mb-3">
                    {advisorData.market_context?.macro_logic || "暂无详细宏观逻辑分析。"}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-xs font-bold border border-indigo-200">
                      状态: {advisorData.market_context?.regime || "未知状态"}
                    </span>
                    <span className="px-3 py-1 bg-pink-100 text-pink-700 rounded-full text-xs font-bold border border-pink-200">
                      主攻: {advisorData.market_context?.target_sectors?.join(", ") || "保持观望"}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

                  <div className="bg-white/80 p-5 rounded-2xl border border-emerald-50 shadow-sm flex flex-col h-full max-h-[350px]">
                    <h3 className="text-sm font-bold text-emerald-700 mb-3 flex items-center gap-2"><PieChart className="w-4 h-4"/> 建议调仓指令</h3>
                    <div className="space-y-4 overflow-y-auto scrollbar-thin pr-2">
                      {advisorData.actions?.existing_adjustments?.map((adj: any, i: number) => (
                        <div key={i} className="flex flex-col border-b border-slate-50 pb-3 gap-1.5">
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-bold text-slate-700 flex items-center">
                              {adj.name}
                              {renderFundClassBadge(adj.name)}
                              <span className="text-xs font-normal text-slate-400 ml-1">({adj.code || "未知"})</span>
                            </span>
                            <span className={`px-2.5 py-1 rounded text-[10px] font-black tracking-wider ml-2 ${
                              adj.action === 'HOLD' ? 'bg-slate-100 text-slate-600' :
                              adj.action === 'REDUCE' || adj.action === 'CLEAR' ? 'bg-rose-100 text-rose-600' : 'bg-emerald-100 text-emerald-600'
                            }`}>
                              {adj.action}
                            </span>
                          </div>
                          <p className="text-xs text-slate-500 leading-relaxed bg-slate-50/50 p-2 rounded-lg border border-slate-100">
                            {adj.reasoning || "AI暂未提供详细调仓说明，建议保持关注。"}
                          </p>
                        </div>
                      ))}
                      {(!advisorData.actions?.existing_adjustments || advisorData.actions.existing_adjustments.length === 0) && (
                        <div className="flex items-center justify-center h-full pt-8">
                          <span className="text-xs text-slate-400">当前您的账户无持仓，无需调仓。</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-violet-500 to-fuchsia-500 p-5 rounded-2xl shadow-md text-white flex flex-col h-full max-h-[350px] overflow-hidden">
                    <h3 className="text-sm font-bold text-white/90 mb-3 flex-shrink-0 flex items-center justify-between">
                      <span>✨ 优质组合推荐</span>
                      <span className="text-[10px] font-normal opacity-70 bg-black/20 px-2 py-0.5 rounded-full">横向滑动查看 ➔</span>
                    </h3>

                    <div className="flex overflow-x-auto gap-3 pb-2 scrollbar-thin scrollbar-thumb-white/30 scrollbar-track-white/5 flex-grow">
                      {advisorData.actions?.new_recommendations?.map((rec: any, i: number) => (
                        <div key={i} className="bg-white/10 border border-white/20 rounded-xl p-4 backdrop-blur-md min-w-[220px] max-w-[220px] flex-shrink-0 flex flex-col transition hover:bg-white/20">
                          <div className="mb-3">
                            <p className="text-sm font-bold mb-0.5 truncate flex items-center" title={rec.name}>
                              <span className="truncate">{rec.name}</span>
                              {renderFundClassBadge(rec.name)}
                            </p>
                            <div className="flex justify-between items-center mt-1">
                              <p className="text-[10px] font-mono opacity-80 bg-black/20 px-1.5 py-0.5 rounded">{rec.code}</p>
                              <p className="text-[10px] opacity-80 text-amber-300 font-bold">建议持有 {rec.holding_period_months} 个月</p>
                            </div>
                          </div>
                          <div className="text-xs bg-black/20 p-2.5 rounded-lg leading-relaxed flex-grow overflow-y-auto scrollbar-none opacity-90 text-white/90 shadow-inner">
                            {rec.reasoning}
                          </div>
                        </div>
                      ))}
                      {(!advisorData.actions?.new_recommendations || advisorData.actions.new_recommendations.length === 0) && (
                        <div className="flex items-center justify-center w-full min-h-[150px]">
                          <span className="text-xs text-white/80 bg-black/10 px-4 py-2 rounded-lg">系统当前判定无较好的开仓机会，建议观望</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ) : selectedDate && adviceHistory.length > 0 ? (
              <div className="py-12 text-center relative z-10 bg-white/40 rounded-2xl border border-dashed border-indigo-200 animate-fade-in">
                <Calendar className="w-12 h-12 mx-auto text-indigo-300 mb-3 opacity-50" />
                <p className="text-sm text-slate-500 font-medium">在 {selectedDate} 这一天，系统没有生成策略报告喵~</p>
                <button
                  onClick={() => fetchAdvisorHistory(true)}
                  className="mt-4 px-5 py-2 bg-indigo-50 text-indigo-600 rounded-full text-xs font-bold hover:bg-indigo-100 transition shadow-sm border border-indigo-100"
                >
                  返回查看最新报告
                </button>
              </div>
            ) : (
              <div className="py-12 text-center relative z-10 bg-white/40 rounded-2xl border border-dashed border-indigo-200">
                <Bot className="w-12 h-12 mx-auto text-indigo-300 mb-3 opacity-50" />
                <p className="text-sm text-slate-500 font-medium">点击右上角按钮，获取基于最新量化数据的定制报告喵~</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}