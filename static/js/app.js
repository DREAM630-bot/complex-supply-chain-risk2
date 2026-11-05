// ============ 供应链风险全景测度系统 - 前端交互逻辑 ============
let matrixData = null, topoData = null, cutsData = null, pipelineData = null;
let matrixChart = null, topoChart = null;

const QUADRANT_COLOR = { dominant_core: '#ef5350', hidden_critical: '#ffa726', local_disturbance: '#29b6f6', routine: '#66bb6a' };
const QUADRANT_NAME = { dominant_core: '显性核心', hidden_critical: '隐性关键', local_disturbance: '局部扰动', routine: '常规节点' };
const TYPE_NAME = { supply: '奶源', factory: '工厂', warehouse: '分仓', delivery: '配送', terminal: '终端' };
const PRECISION_NAME = { absolute_anchor: '绝对锚点', proxy_ratio: '代理比率', prior_assumption: '先验假设' };

async function fetchJSON(url, opts) { const r = await fetch(url, opts); return r.json(); }

// ============ 初始化 ============
async function init() {
  try {
    let st;
    for (let i = 0; i < 120; i++) {
      st = await fetchJSON('/api/status');
      if (st.ready) break;
      await new Promise(r => setTimeout(r, 1500));
    }
    if (!st.ready) throw new Error('预计算超时');
    document.getElementById('statusBadge').textContent = '系统就绪';
    document.getElementById('statusBadge').classList.remove('loading');
    document.getElementById('loadingOverlay').style.display = 'none';

    const [matrix, topo, cuts, pipeline] = await Promise.all([
      fetchJSON('/api/matrix'), fetchJSON('/api/topology'),
      fetchJSON('/api/cuts'), fetchJSON('/api/pipeline')
    ]);
    matrixData = matrix; topoData = topo; cutsData = cuts; pipelineData = pipeline;

    renderMatrix(); renderTopo(); renderCuts(); renderPipeline(); renderRanking();
    const cv = matrix.cross_validation;
    if (cv && cv.pearson !== null && cv.pearson !== undefined) {
      document.getElementById('crossValInfo').textContent =
        '交叉验证 Pearson r=' + cv.pearson + '（低相关印证二维矩阵必要性） | 采样' + matrix.n_samples + '次';
    }
  } catch (e) {
    console.error(e);
    document.getElementById('loadingOverlay').innerHTML =
      '<div style="color:#ef5350;font-size:15px">加载失败：' + e.message + '<br>请确认后端服务已启动（python app.py）</div>';
  }
}

// ============ 二维矩阵散点图 ============
function renderMatrix() {
  matrixChart = echarts.init(document.getElementById('matrixChart'));
  const m = matrixData;
  const groups = {};
  m.matrix.forEach(n => {
    (groups[n.quadrant] = groups[n.quadrant] || []).push({
      value: [n.exposure, n.responsibility], name: n.name, nodeId: n.node_id,
      symbolSize: n.is_cut_point ? 24 : 14,
      emphasis: { label: { show: true, formatter: '{b}' } }
    });
  });
  const series = Object.keys(groups).map(q => ({
    name: QUADRANT_NAME[q], type: 'scatter', data: groups[q],
    itemStyle: { color: QUADRANT_COLOR[q], shadowBlur: 6, shadowColor: 'rgba(0,0,0,0.4)' },
    label: { show: true, formatter: p => p.data.nodeId, position: 'top', fontSize: 10, color: '#b0bec5' }
  }));
  series.push({
    type: 'scatter', silent: true, data: [],
    markLine: {
      silent: true, symbol: 'none', lineStyle: { type: 'dashed', color: '#546e7a' },
      label: { color: '#78909c', fontSize: 11 },
      data: [
        { xAxis: m.exp_threshold, label: { formatter: '暴露度阈值 ' + m.exp_threshold.toFixed(2) } },
        { yAxis: m.resp_threshold, label: { formatter: '责任度阈值 ' + m.resp_threshold.toFixed(3) } }
      ]
    }
  });
  matrixChart.setOption({
    tooltip: {
      trigger: 'item', backgroundColor: '#1e2d3d', borderColor: '#37474f', textStyle: { color: '#cfd8dc', fontSize: 12 },
      formatter: p => {
        if (!p.data.nodeId) return '';
        const n = m.matrix.find(x => x.node_id === p.data.nodeId);
        return '<b>' + n.name + '（' + n.node_id + '）</b><br>类型：' + TYPE_NAME[n.type] +
          '<br>暴露度：' + n.exposure + '<br>责任度：' + n.responsibility +
          '<br>象限：' + QUADRANT_NAME[n.quadrant] + (n.is_cut_point ? '<br><span style="color:#ef5350">⚠ 割点</span>' : '') +
          '<br>数据精度：' + (PRECISION_NAME[n.capacity_source] || n.capacity_source) +
          '<br><i style="color:#4fc3f7">点击可让智能体解读</i>';
      }
    },
    legend: { data: Object.keys(groups).map(q => QUADRANT_NAME[q]), textStyle: { color: '#90a4ae', fontSize: 11 }, top: 0 },
    grid: { left: 60, right: 30, top: 40, bottom: 50 },
    xAxis: { name: '暴露度（文本+气象）', nameTextStyle: { color: '#78909c' }, min: 0, max: 1, axisLine: { lineStyle: { color: '#37474f' } }, axisLabel: { color: '#78909c' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
    yAxis: { name: '责任度（Shapley）', nameTextStyle: { color: '#78909c' }, min: 0, axisLine: { lineStyle: { color: '#37474f' } }, axisLabel: { color: '#78909c' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
    series: series
  });
  matrixChart.on('click', p => { if (p.data && p.data.nodeId) askQuestion('请解释' + p.data.nodeId + '的风险'); });
}

// ============ 拓扑网络图 ============
function renderTopo() {
  topoChart = echarts.init(document.getElementById('topoChart'));
  const t = topoData;
  const nodes = t.nodes.map(n => ({
    id: n.id, name: n.id + ' ' + n.name,
    symbolSize: Math.max(18, 18 + n.shapley * 150),
    itemStyle: { color: QUADRANT_COLOR[n.quadrant], borderColor: t.sources.includes(n.id) ? '#ffeb3b' : (t.sinks.includes(n.id) ? '#ce93d8' : '#1e3a5f'), borderWidth: t.sources.includes(n.id) || t.sinks.includes(n.id) ? 3 : 1 },
    attributes: n
  }));
  const links = t.edges.map(e => ({
    source: e.from, target: e.to, value: e.capacity,
    lineStyle: { width: Math.max(1, e.capacity * 4), color: e.bidirectional ? '#ffa726' : '#4a6a8a', curveness: e.bidirectional ? 0.2 : 0 }
  }));
  topoChart.setOption({
    tooltip: {
      backgroundColor: '#1e2d3d', borderColor: '#37474f', textStyle: { color: '#cfd8dc', fontSize: 12 },
      formatter: p => {
        if (p.dataType === 'edge') {
          const e = t.edges.find(x => x.from === p.data.source && x.to === p.data.target);
          return p.data.source + ' → ' + p.data.target + '<br>容量：' + e.capacity + '<br>模式：' + e.mode + (e.bidirectional ? '<br>（双向调拨）' : '');
        }
        const n = p.data.attributes;
        return '<b>' + n.name + '（' + n.id + '）</b><br>类型：' + TYPE_NAME[n.type] +
          '<br>容量：' + n.capacity + '（' + (PRECISION_NAME[n.capacity_source] || n.capacity_source) + '）' +
          '<br>Shapley：' + n.shapley + '<br>暴露度：' + n.exposure + '<br>象限：' + QUADRANT_NAME[n.quadrant];
      }
    },
    series: [{
      type: 'graph', layout: 'force', roam: true, draggable: true,
      data: nodes, links: links,
      force: { repulsion: 400, edgeLength: [60, 140], gravity: 0.08 },
      edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 7,
      label: { show: true, fontSize: 10, color: '#cfd8dc', position: 'bottom' },
      lineStyle: { opacity: 0.7 },
      emphasis: { focus: 'adjacency', lineStyle: { width: 5 } }
    }]
  });
  topoChart.on('click', p => { if (p.dataType === 'node') askQuestion('请解释' + p.data.attributes.id + '的风险'); });
}

// ============ 割集分析 ============
function renderCuts() {
  const c = cutsData.cuts, cand = cutsData.candidates;
  const nodeMap = {};
  topoData.nodes.forEach(n => nodeMap[n.id] = n.name);
  let html = '<div class="info-grid">';
  html += '<div class="info-card"><h4>全局最小割容量</h4><p style="font-size:22px;color:#4fc3f7">' + c.mincut_value.toFixed(2) + '</p><p>超级源S→超级汇T的最小割，即全局瓶颈容量上限。木桶效应数学化：系统最大通过能力由最窄割集决定。</p></div>';
  html += '<div class="info-card"><h4>割点（结构性命门）</h4><p>' + (c.cut_nodes.length ? c.cut_nodes.map(n => '<span class="tag cut">' + n + ' ' + (nodeMap[n] || '') + '</span>').join(' ') : '无') + '</p><p>一旦失效将分割网络，部分需求汇失去供应通路。</p></div>';
  html += '<div class="info-card"><h4>瓶颈割集边</h4>' + (c.bottleneck_edges && c.bottleneck_edges.length ? '<table><tr><th>边</th><th>容量</th><th>模式</th></tr>' + c.bottleneck_edges.map(e => '<tr><td>' + e.from + '→' + e.to + '</td><td>' + e.capacity.toFixed(2) + '</td><td>' + e.mode + '</td></tr>').join('') + '</table>' : '<p>无瓶颈边数据</p>') + '</div>';
  html += '<div class="info-card"><h4>关键候选集（分层Shapley）</h4><p>割点：' + (cand.cut_nodes.join(', ') || '无') + '</p><p>桥端点：' + (cand.bridge_ends.join(', ') || '无') + '</p><p>高吞吐：' + cand.high_throughput.join(', ') + '</p><p>关键转换点：' + (cand.key_converters.join(', ') || '无') + '</p></div>';
  html += '</div>';
  if (c.bridges && c.bridges.length) {
    html += '<h4 style="color:#4fc3f7;margin:14px 0 8px">桥边（唯一连接通路）</h4><p style="font-size:12px;color:#90a4ae">' + c.bridges.map(b => b.join('↔')).join('，') + '</p>';
  }
  document.getElementById('cutsContent').innerHTML = html;
}

// ============ 流水线审计 ============
function renderPipeline() {
  const p = pipelineData;
  let html = '';
  p.event_results.forEach(ev => {
    const rejected = ev.status === 'rejected_by_R1';
    html += '<div class="audit-item" style="' + (rejected ? 'border-color:#c62828;opacity:0.7' : '') + '">';
    html += '<span class="ev-id">' + ev.event_id + '</span> ';
    html += '<span style="color:#90a4ae">' + (ev.date || '') + ' | ' + (ev.source || '') + ' | 范围词:' + (ev.scope_word || '') + ' | 时效:' + (ev.time_decay_flag || '') + '</span><br>';
    html += '<span style="color:#b0bec5">节点：' + (ev.node_ids || []).join(', ') + '</span><br>';
    if (rejected) {
      html += '<span style="color:#ef5350">⛔ R1硬门禁驳回：证据锚点未在原文逐字存在，该事件被剔除</span>';
    } else {
      html += 'severity_raw: <b style="color:#4fc3f7">' + ev.severity_raw + '</b> → score_delta: <b style="color:' + (ev.score_delta < 0 ? '#ef5350' : '#66bb6a') + '">' + ev.score_delta + '</b> → severity_final: <b style="color:#ffa726">' + ev.severity_final + '</b><br>';
      html += '证据锚点：「<span style="color:#80d8ff">' + ev.evidence_anchor + '</span>」<br>';
      (ev.audit_trail || []).forEach(rule => {
        html += '<span class="rule ' + (rule.passed ? 'pass' : 'fail') + '">' + rule.rule + (rule.delta !== 0 ? ' Δ' + rule.delta : '') + '：' + rule.detail + '</span>';
      });
    }
    html += '</div>';
  });
  document.getElementById('pipelineContent').innerHTML = html;
}

// ============ 责任度排序 ============
function renderRanking() {
  const rank = matrixData.shapley_rank;
  let html = '<table><tr><th>排名</th><th>节点</th><th>责任度(Shapley)</th><th>象限</th><th>暴露度</th><th>割点</th></tr>';
  rank.forEach((r, i) => {
    const m = matrixData.matrix.find(x => x.node_id === r.node_id);
    html += '<tr><td>' + (i + 1) + '</td><td>' + r.node_id + ' ' + (m ? m.name : '') + '</td><td style="color:#4fc3f7;font-weight:600">' + r.value + '</td><td><span class="tag ' + (m ? m.quadrant : '') + '">' + QUADRANT_NAME[m ? m.quadrant : ''] + '</span></td><td>' + (m ? m.exposure : '') + '</td><td>' + (m && m.is_cut_point ? '<span class="tag cut">割点</span>' : '') + '</td></tr>';
  });
  html += '</table>';
  html += '<p style="font-size:12px;color:#78909c;margin-top:10px">责任度=节点失效对全局供应保障的平均边际破坏力（蒙特卡罗排列采样' + matrixData.n_samples + '次）。Top节点失效时最大流与连通性下降最剧烈。</p>';
  document.getElementById('rankingContent').innerHTML = html;
}

// ============ 敏感性分析 ============
async function runSensitivity() {
  const btn = event.target;
  btn.disabled = true; btn.textContent = '计算中...（约30-60秒）';
  const resultEl = document.getElementById('sensitivityResult');
  resultEl.innerHTML = '<div style="color:#4fc3f7;font-size:13px">正在执行±25%容量扰动扫描与Kendall\'s W对称性诊断...</div>';
  try {
    const data = await fetchJSON('/api/sensitivity');
    const s = data.sensitivity, sym = data.symmetry;
    let html = '<div class="info-grid">';
    html += '<div class="info-card"><h4>Kendall\'s W（扰动下排序一致性）</h4><p style="font-size:22px;color:' + (s.kendalls_w > 0.7 ? '#66bb6a' : '#ffa726') + '">' + (s.kendalls_w !== null && s.kendalls_w !== undefined ? s.kendalls_w.toFixed(4) : 'N/A') + '</p><p>W>0.7表示扰动下Shapley排序高度一致，核心结论稳健。</p></div>';
    html += '<div class="info-card"><h4>Spearman秩相关均值</h4><p style="font-size:22px;color:#4fc3f7">' + (s.spearman_mean !== null && s.spearman_mean !== undefined ? s.spearman_mean.toFixed(4) : 'N/A') + '</p><p>基线排序与扰动排序的秩相关，越接近1越稳健。</p></div>';
    html += '<div class="info-card"><h4>Top3稳定性</h4><p style="font-size:15px;color:' + (s.top3_stable ? '#66bb6a' : '#ffa726') + '">' + (s.top3_stable ? '✓ 稳定' : '✗ 不稳定') + '</p><p>扰动后责任度前三名是否保持不变。</p></div>';
    html += '<div class="info-card"><h4>对称性诊断（纯最大流Shapley）</h4><p style="font-size:13px;color:#90a4ae">' + (sym.conclusion || sym.note || JSON.stringify(sym)) + '</p><p style="font-size:12px;color:#78909c">Kendall\'s W=' + (sym.kendalls_w !== undefined ? sym.kendalls_w : 'N/A') + '：纯最大流特征函数对同结构节点区分度低，印证融合连通性惩罚的必要性。</p></div>';
    html += '</div>';
    if (s.conclusion) html += '<p style="font-size:12px;color:#78909c;margin-top:12px">' + s.conclusion + '</p>';
    if (s.trials && s.trials.length) {
      html += '<h4 style="color:#4fc3f7;margin:14px 0 8px">扰动试次明细</h4><table><tr><th>试次</th><th>扰动幅度</th><th>Kendall\'s W</th><th>Spearman</th></tr>';
      s.trials.forEach((t, i) => {
        html += '<tr><td>' + (i + 1) + '</td><td>±' + Math.round((t.perturb || 0) * 100) + '%</td><td>' + (t.kendalls_w !== undefined ? t.kendalls_w.toFixed(4) : '-') + '</td><td>' + (t.spearman !== undefined ? t.spearman.toFixed(4) : '-') + '</td></tr>';
      });
      html += '</table>';
    }
    resultEl.innerHTML = html;
  } catch (e) {
    resultEl.innerHTML = '<div style="color:#ef5350;font-size:13px">计算失败：' + e.message + '</div>';
  }
  btn.disabled = false; btn.textContent = '运行敏感性分析';
}

// ============ 智能体聊天 ============
function addMsg(text, isUser, isOutOfScope, meta) {
  const box = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg ' + (isUser ? 'user' : 'bot') + (isOutOfScope ? ' out-of-scope' : '');
  div.innerHTML = text.replace(/\n/g, '<br>');
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    div.appendChild(m);
  }
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function askQuestion(q) {
  document.getElementById('chatInput').value = q;
  sendChat();
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  addMsg(q, true);
  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'msg bot';
  loadingDiv.textContent = '思考中...';
  loadingDiv.id = 'thinking';
  document.getElementById('chatMessages').appendChild(loadingDiv);
  document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
  try {
    const resp = await fetchJSON('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    loadingDiv.remove();
    const meta = resp.engine === 'deepseek_api' ? '[DeepSeek API · 强约束JSON Context]' : '[规则模板引擎 · 可验证可复现]';
    addMsg(resp.reply, false, !resp.in_scope, meta);
  } catch (e) {
    loadingDiv.remove();
    addMsg('通信失败：' + e.message, false, true);
  }
}

// ============ Tab切换 ============
function switchTab(name, btn) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-headers button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'topo' && topoChart) topoChart.resize();
  if (name === 'matrix') { /* noop */ }
}

window.addEventListener('resize', () => {
  if (matrixChart) matrixChart.resize();
  if (topoChart) topoChart.resize();
});

init();