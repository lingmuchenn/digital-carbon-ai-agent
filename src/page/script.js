let isExpanded = false;
let currentStep = 1;
let selectedMode = 'similar'; // 'similar', 'duplicate', 'process', 'history'
let analysisResults = [];
let currentCardIndex = 0;
let isAnalyzing = false; // 标记是否正在分析
let activeEventSource = null; // 保存活跃的 SSE 连接
let progressTimer = null; // 进度条定时器
let progressStartTime = null; // 进度开始时间
let estimatedDuration = 0; // 预估总时长（秒）
let lastAnalyzedPath = ''; // 记录本次分析的根目录（用于安全删除）

// ==== 悬浮球展开/收起 ====
function toggleExpand(event) {
    if (event) event.stopPropagation();
    const ball = document.getElementById('floating-ball');
    const box = document.getElementById('interaction-box');
    const layer = document.getElementById('step1-layer');
    const centerIcon = document.querySelector('.step-1-center-icon');
    isExpanded = !isExpanded;
    if (isExpanded) {
        box.classList.remove('hidden');
        box.style.pointerEvents = 'auto';
        // 展开时默认展示功能球层
        if (layer) layer.classList.remove('hidden');
        if (centerIcon) centerIcon.classList.remove('hidden');
        
        // 只有在 Step 1 时才隐藏原 Logo
        if (currentStep === 1) {
            if (ball) ball.style.visibility = 'hidden';
        }
        
        showStep(currentStep);
        // step1 视觉展开
        if (currentStep === 1) box.classList.add('step1-open');
    } else {
        box.classList.add('hidden');
        box.style.pointerEvents = 'none';
        if (layer) layer.classList.add('hidden');
        
        // 恢复初始 Logo
        if (ball) ball.style.visibility = 'visible';
        
        box.classList.remove('step1-open');
    }
}

// 点击外部关闭（只隐藏，不中断）
document.addEventListener('click', (event) => {
    if (!isExpanded) return;
    const ball = document.getElementById('floating-ball');
    const box = document.getElementById('interaction-box');
    
    // 如果点击的是球本身，由 toggleExpand 处理，这里直接跳过
    if (ball.contains(event.target)) return;
    
    // 如果点击的是容器外部，则关闭
    if (!box.contains(event.target)) {
        isExpanded = false;
        box.classList.add('hidden');
        box.classList.remove('step1-open');
        ball.style.visibility = 'visible';
    }
});

// 阻止容器内部点击冒泡到 document，防止触发关闭
document.getElementById('interaction-box').addEventListener('click', (event) => {
    event.stopPropagation();
});

// ==== 步骤切换 ====
function showStep(step) {
    currentStep = step;
    const layer = document.getElementById('step1-layer');
    const box = document.getElementById('interaction-box');
    const ball = document.getElementById('floating-ball');

    if (layer) layer.classList.toggle('hidden', step !== 1);
    
    if (step === 1) {
        if (box && !box.classList.contains('hidden')) {
            box.classList.add('step1-open');
        }
        if (ball && isExpanded) {
            ball.style.visibility = 'hidden';
        }
    } else {
        if (box) box.classList.remove('step1-open');
        if (ball) {
            ball.style.visibility = 'visible';
        }
    }

    for (let i = 2; i <= 4; i++) {
        const el = document.getElementById(`step-${i}`);
        if (el) el.classList.toggle('hidden', i !== step);
    }
}

// ==== STEP 1 → STEP 2: 选择模式并跳转到文件夹选择界面 ====
function selectModeAndGoStep2(mode) {
    selectedMode = mode;
    const labels = { 
        'similar': '相似文件', 
        'duplicate': '重复文件', 
        'process': '过程文件',
        'history': '历史文件'
    };
    
    // 过程文件和历史文件暂时不接入后端功能
    if (mode === 'process' || mode === 'history') {
        alert(`${labels[mode]}功能即将上线，敬请期待！`);
        return;
    }
    
    // 跳转到新的文件夹选择界面（Step 2）
    showStep(2);
    
    // 聚焦到路径输入框
    setTimeout(() => {
        const newPathInput = document.getElementById('folder-path-input');
        if (newPathInput) newPathInput.focus();
    }, 100);
}

// ==== 关闭 Step 1（返回按钮） ====
function closeStep1() {
    isExpanded = false;
    const box = document.getElementById('interaction-box');
    const ball = document.getElementById('floating-ball');
    if (box) box.classList.add('hidden');
    if (ball) {
        ball.style.visibility = 'visible';
    }
}

// ==== 从第二步返回第一步 ====
function goBackToStep1() {
    showStep(1);
    // 清空路径输入
    document.getElementById('path-input').value = '';
}

// ==== STEP 2: 输入路径并开始分析 ====
function startAnalysisFromPath() {
    const pathInput = document.getElementById('path-input');
    const folderPath = pathInput.value.trim();
    
    if (!folderPath) {
        alert('请输入文件夹路径');
        return;
    }
    
    console.log('开始分析文件夹:', folderPath);
    goToStep3(folderPath);
}

// ==== STEP 2 → STEP 3: 开始后端分析 ====
function goToStep3(folderPath) {
    showStep(3);
    isAnalyzing = true; // 标记开始分析
    lastAnalyzedPath = folderPath;
    
    // 重置进度条
    resetProgress();
    
    const logContainer = document.getElementById('log-container');
    logContainer.innerHTML = '';
    // 添加初始化日志
    addLogLine('🚀 正在初始化分析引擎...');
    
    // 启动后端分析并建立 SSE 连接
    startBackendAnalysis(folderPath);
}

// 重置进度条
function resetProgress() {
    const progressWrapper = document.getElementById('progress-wrapper');
    const progressFill = document.getElementById('progress-fill');
    const progressTime = document.getElementById('progress-time');
    const progressHint = document.getElementById('progress-hint');
    
    progressWrapper.style.display = 'none';
    progressFill.style.width = '0%';
    progressTime.textContent = '0%';
    progressHint.textContent = '正在处理中...';
    
    // 清除旧的定时器
    if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
    }
    progressStartTime = null;
    estimatedDuration = 0;
}

// 启动进度条
function startProgress(totalSeconds, totalFiles) {
    const progressWrapper = document.getElementById('progress-wrapper');
    const progressFill = document.getElementById('progress-fill');
    const progressTime = document.getElementById('progress-time');
    const progressHint = document.getElementById('progress-hint');
    
    progressWrapper.style.display = 'block';
    progressStartTime = Date.now();
    estimatedDuration = totalSeconds;
    
    // 更新提示文本
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    let timeStr = '';
    if (minutes > 0) {
        timeStr = `${minutes}分${seconds}秒`;
    } else {
        timeStr = `${seconds}秒`;
    }
    progressHint.textContent = `预计时间: ${timeStr} (共 ${totalFiles} 个文件)`;
    
    // 每500ms更新一次进度
    progressTimer = setInterval(() => {
        const elapsed = (Date.now() - progressStartTime) / 1000; // 已过时间（秒）
        let progress = Math.min((elapsed / estimatedDuration) * 100, 99); // 最多到99%
        
        progressFill.style.width = `${progress}%`;
        progressTime.textContent = `${Math.round(progress)}%`;
        
        // 如果超过预估时间，提示用户
        if (elapsed > estimatedDuration) {
            progressHint.textContent = '处理时间超出预期，请稍候...';
        }
    }, 500);
}

// 完成进度
function completeProgress() {
    const progressFill = document.getElementById('progress-fill');
    const progressTime = document.getElementById('progress-time');
    const progressHint = document.getElementById('progress-hint');
    
    if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
    }
    
    progressFill.style.width = '100%';
    progressTime.textContent = '100%';
    progressHint.textContent = '✅ 分析完成！';
}

// 启动后端分析
async function startBackendAnalysis(folderPath) {
    try {
        console.log('发送分析请求:', { path: folderPath, mode: selectedMode });
        
        // 1. 启动分析任务
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                path: folderPath, 
                mode: selectedMode 
            })
        });
        
        console.log('分析请求响应状态:', response.status);
        
        if (!response.ok) {
            const errorData = await response.json();
            console.error('分析请求失败:', errorData);
            throw new Error(errorData.error || '启动分析失败');
        }
        
        // 2. 建立 SSE 连接接收实时日志
        if (activeEventSource) {
            activeEventSource.close();
        }
        
        const eventSource = new EventSource('/api/stream');
        activeEventSource = eventSource; // 保存连接引用
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'log') {
                addLogLine(data.message);
            } else if (data.type === 'reset') {
                // 后端开始新任务时的重置事件
            } else if (data.type === 'estimate') {
                // 显示预估时间并启动进度条
                const seconds = data.seconds;
                const total = data.total_files;
                let timeStr = seconds < 60 ? `${seconds}秒` : `${Math.ceil(seconds/60)}分钟`;
                addLogLine(`⏱️ 预计耗时: ${timeStr} (共 ${total} 个文件)`);
                
                // 启动进度条
                startProgress(seconds, total);
            } else if (data.type === 'estimate_update') {
                // 动态追加预计时间
                const add = Number(data.add_seconds || 0);
                if (add > 0 && estimatedDuration > 0) {
                    estimatedDuration += add;
                    addLogLine(`⏱️ 预计时间更新：+${add}秒（${data.reason || 'update'}）`);
                    // 更新提示文案
                    const progressHint = document.getElementById('progress-hint');
                    const minutes = Math.floor(estimatedDuration / 60);
                    const seconds = Math.floor(estimatedDuration % 60);
                    const timeStr = minutes > 0 ? `${minutes}分${seconds}秒` : `${seconds}秒`;
                    progressHint.textContent = `预计时间: ${timeStr}`;
                }
            } else if (data.type === 'result') {
                console.log('✅ 收到结果数据，关闭连接');
                completeProgress();
                safeCloseEventSource();
                isAnalyzing = false;
                analysisResults = formatResults(data.data, data.mode);
                
                if (analysisResults.length === 0) {
                    addLogLine('✅ 分析完成，未发现需要处理的文件');
                    setTimeout(() => {
                        alert('未发现重复或相似文件');
                        resetProgress();
                        showStep(1);
                    }, 1000);
                } else {
                    addLogLine(`✅ 分析完成，即将跳转结果页 (${analysisResults.length}组)...`);
                    setTimeout(() => showStep4(), 1000);
                }
            } else if (data.type === 'error') {
                addLogLine(`❌ 错误: ${data.message}`);
                if (progressTimer) {
                    clearInterval(progressTimer);
                    progressTimer = null;
                }
                safeCloseEventSource();
                isAnalyzing = false;
            } else if (data.type === 'ping') {
                // keepalive
            }
        };
        
        eventSource.onerror = (error) => {
            if (!isAnalyzing) return;
            console.error('SSE 连接错误:', error);
            if (eventSource.readyState === EventSource.CLOSED) {
                 addLogLine('⚠️ 连接已断开 (请检查后台是否仍在运行)');
                 if (progressTimer) {
                     clearInterval(progressTimer);
                     progressTimer = null;
                 }
                 safeCloseEventSource();
                 isAnalyzing = false;
            }
        };
        
    } catch (error) {
        console.error('分析启动失败:', error);
        addLogLine(`❌ 启动失败: ${error.message}`);
        isAnalyzing = false;
    }
}

function safeCloseEventSource() {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }
}

// 格式化后端结果
function formatResults(data, mode) {
    if (!data || !Array.isArray(data)) return [];
    
    return data.map(item => {
        if (!item) return null;

        // --- 核心原则：不修改重复文件原有的判断逻辑 ---
        if (mode === 'duplicate' || item.type === 'duplicate' || (!item.type && item.files)) {
             return {
                type: 'duplicate',
                files: item.files || [],
                label: '重复文件组',
                fileSize: item.file_size_mb || 0,
                analysis: item.analysis,
                needCleanup: item.need_cleanup
            };
        }

        // --- 相似文件特有逻辑 ---
        if (mode === 'similar' || item.type === 'similar' || item.type === 'photo_group' || 
            (item.type && item.type.includes('screenshot'))) {
            
            // 1. 处理两两相似 (Pairwise) -> 转换为组形式以沿用重复文件卡片
            if (item.file1 && item.file2) {
                // 确保 file1 和 file2 都有 path，如果是对象则取 path，如果是字符串则作为 path
                const f1 = typeof item.file1 === 'object' ? item.file1 : { path: item.file1, name: item.file1.split(/[\\/]/).pop() };
                const f2 = typeof item.file2 === 'object' ? item.file2 : { path: item.file2, name: item.file2.split(/[\\/]/).pop() };
                
                // 设置初始建议：较大的建议删除，较小的保留（或根据 LLM，但这里先做简单兼容）
                if (!f1.suggestion) f1.suggestion = '保留';
                if (!f2.suggestion) f2.suggestion = '删除';

                const s1 = f1.size || f1.file_size || 0;
                const s2 = f2.size || f2.file_size || 0;

                return {
                    type: 'duplicate', // 强制设为 duplicate 以沿用列表渲染
                    files: [f1, f2],
                    label: '相似文件组',
                    similarity: item.similarity || 0,
                    analysis: item.analysis,
                    fileSize: (s1 + s2) / (1024 * 1024)
                };
            }

            // 2. 处理组形式 (照片组、截图组等)
            if (item.files && Array.isArray(item.files)) {
                let label = '相似文件组';
                let cardType = 'duplicate'; // 借用列表渲染
                
                if (item.type === 'photo_group') {
                    label = '📸 相似照片组 (保留最佳)';
                    cardType = 'photo_group';
                } else if (item.type === 'screenshot_dedup_group') {
                    label = '📱 相似截图组 (保留最干净)';
                    cardType = 'screenshot_group';
                } else if (item.type === 'screenshot_category') {
                    label = `📱 截图分类: ${item.label || '其他'}`;
                    cardType = 'screenshot_category';
                }

                // 统一改为 duplicate 类型以进入通用渲染流程
                return {
                    type: 'duplicate', 
                    files: item.files,
                    label: label,
                    bestShot: item.best_shot,
                    category: item.category,
                    groupId: item.group_id,
                    fileSize: item.file_size_mb || 0,
                    needCleanup: item.need_cleanup,
                    analysis: item.analysis || (item.type === 'screenshot_category' ? `检测到 ${item.files.length} 张属于“${item.label}”分类的截图。` : null)
                };
            }
        }
        
        return null;
    }).filter(item => item !== null);
}

function addLogLine(text) {
    const logContainer = document.getElementById('log-container');
    if (!logContainer) return;

    const line = document.createElement('div');
    line.className = 'log-line';
    line.textContent = text;
    
    logContainer.appendChild(line);
    logContainer.scrollTop = logContainer.scrollHeight;
    
    const lines = logContainer.querySelectorAll('.log-line');
    if (lines.length > 100) {
        lines[0].remove();
    }
}

// ==== STEP 4: 展示卡片结果 ====
function showStep4() {
    showStep(4);
    isAnalyzing = false;
    currentCardIndex = 0;
    renderStackedCards();
}

// 根据文件大小更新脱碳等级 UI
function updateCarbonLevel(totalSizeMB) {
    const step4 = document.getElementById('step-4');
    if (!step4) return;

    // 移除旧的等级类
    step4.classList.remove('level-a', 'level-b', 'level-c', 'level-d', 'level-e');

    let level = 'e';
    let emoji = '🤯';
    let levelName = 'E级';
    let levelDesc = '极高碳负担';

    if (totalSizeMB < 1) {
        level = 'a';
        emoji = '🙂';
        levelName = 'A级';
        levelDesc = '低碳负担';
    } else if (totalSizeMB < 5) {
        level = 'b';
        emoji = '😐';
        levelName = 'B级';
        levelDesc = '轻度碳负担';
    } else if (totalSizeMB < 10) {
        level = 'c';
        emoji = '😧';
        levelName = 'C级';
        levelDesc = '中度碳负担';
    } else if (totalSizeMB < 15) {
        level = 'd';
        emoji = '🥺';
        levelName = 'D级';
        levelDesc = '高碳负担';
    }

    step4.classList.add(`level-${level}`);
    
    // 更新头部文字
    const emojiEl = step4.querySelector('.level-emoji');
    const textEl = step4.querySelector('.level-text');
    const descEl = step4.querySelector('.level-desc');
    
    if (emojiEl) emojiEl.textContent = emoji;
    if (textEl) textEl.textContent = levelName;
    if (descEl) descEl.textContent = levelDesc;
}

// MacOS 风格的文件名缩略逻辑
function formatFileName(name, maxLength = 20) {
    if (!name || name.length <= maxLength) return name;
    
    // 提取扩展名
    const lastDot = name.lastIndexOf('.');
    let ext = "";
    let base = name;
    if (lastDot !== -1 && name.length - lastDot <= 7) {
        ext = name.substring(lastDot);
        base = name.substring(0, lastDot);
    }
    
    const targetBaseLength = maxLength - 3 - ext.length; // 3 是省略号的长度
    if (targetBaseLength <= 2) return name; // 太短了就不缩略了
    
    // 寻找分割点：优先在 _ - 空格 处分割 (在 40%-60% 范围内寻找)
    const startIdx = Math.floor(base.length * 0.4);
    const endIdx = Math.ceil(base.length * 0.7);
    const middlePart = base.substring(startIdx, endIdx);
    
    const splitMatch = middlePart.match(/[_\-\s]/);
    if (splitMatch) {
        const splitPos = startIdx + splitMatch.index;
        return base.substring(0, splitPos) + "..." + base.substring(splitPos + 1) + ext;
    }
    
    // 如果没找到符号，按比例缩略 (前 60% 后 40%)
    const frontLen = Math.floor(targetBaseLength * 0.6);
    const backLen = targetBaseLength - frontLen;
    return base.substring(0, frontLen) + "..." + base.substring(base.length - backLen) + ext;
}

// 格式化路径：显示开头和末尾，尽量多展示末尾层级
function formatPath(path) {
    if (!path) return '';
    const normalized = path.replace(/\\/g, '/');
    const parts = normalized.split('/');
    
    // 如果路径长度小于40，直接返回
    if (path.length <= 40) return path;
    
    const drive = parts[0] + (path.includes(':') ? '\\' : '/');
    const fileName = parts[parts.length - 1];
    
    if (parts.length <= 2) return path;

    // 尽量多展示末尾层级，确保显示父文件夹
    if (parts.length >= 5) {
        // 展示 开头 + ... + 倒数第四级 + 倒数第三级 + 倒数第二级
        return `${drive}.../${parts[parts.length-4]}/${parts[parts.length-3]}/${parts[parts.length-2]}`;
    } else if (parts.length >= 4) {
        // 展示 开头 + ... + 倒数第三级 + 倒数第二级
        return `${drive}.../${parts[parts.length-3]}/${parts[parts.length-2]}`;
    }
    
    return path;
}




// 格式化文件大小
function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// 获取文件对应的本地 SVG 图标路径
function getFileIcon(fileName) {
    const ext = fileName.split('.').pop().toLowerCase();
    const iconMap = {
    '3g2': 'image/icon/type=3G2.svg',
    '3ga': 'image/icon/type=3GA.svg',
    '3gp-1': 'image/icon/type=3GP-1.svg',
    '3gp': 'image/icon/type=3GP.svg',
    '7z': 'image/icon/type=7Z.svg',
    'aa': 'image/icon/type=AA.svg',
    'acc': 'image/icon/type=ACC.svg',
    'adn': 'image/icon/type=ADN.svg',
    'aep': 'image/icon/type=AEP.svg',
    'ai': 'image/icon/type=AI.svg',
    'aif': 'image/icon/type=AIF.svg',
    'aifc': 'image/icon/type=AIFC.svg',
    'aiff': 'image/icon/type=AIFF.svg',
    'ait': 'image/icon/type=AIT.svg',
    'amr': 'image/icon/type=AMR.svg',
    'ani': 'image/icon/type=ANI.svg',
    'apk': 'image/icon/type=APK.svg',
    'app': 'image/icon/type=APP.svg',
    'asax': 'image/icon/type=ASAX.svg',
    'asc': 'image/icon/type=ASC.svg',
    'ascx': 'image/icon/type=ASCX.svg',
    'asf': 'image/icon/type=ASF.svg',
    'ash': 'image/icon/type=ASH.svg',
    'ashx': 'image/icon/type=ASHX.svg',
    'asmx': 'image/icon/type=ASMX.svg',
    'asp': 'image/icon/type=ASP.svg',
    'aspx': 'image/icon/type=ASPX.svg',
    'asx': 'image/icon/type=ASX.svg',
    'au': 'image/icon/type=AU.svg',
    'aup': 'image/icon/type=AUP.svg',
    'avi': 'image/icon/type=AVI.svg',
    'axd': 'image/icon/type=AXD.svg',
    'aze': 'image/icon/type=AZE.svg',
    'bak': 'image/icon/type=BAK.svg',
    'bash': 'image/icon/type=BASH.svg',
    'bat': 'image/icon/type=BAT.svg',
    'bin': 'image/icon/type=BIN.svg',
    'blank': 'image/icon/type=BLANK.svg',
    'bmp-1': 'image/icon/type=BMP-1.svg',
    'bmp': 'image/icon/type=BMP.svg',
    'bpg': 'image/icon/type=BPG.svg',
    'browser': 'image/icon/type=BROWSER.svg',
    'bz2': 'image/icon/type=BZ2.svg',
    'c': 'image/icon/type=C.svg',
    'cab': 'image/icon/type=CAB.svg',
    'caf': 'image/icon/type=CAF.svg',
    'cal': 'image/icon/type=CAL.svg',
    'cd': 'image/icon/type=CD.svg',
    'cdr': 'image/icon/type=CDR.svg',
    'cer': 'image/icon/type=CER.svg',
    'css': 'image/icon/type=CSS.svg',
    'csv': 'image/icon/type=CSV.svg',
    'default': 'image/icon/type=DEFAULT.svg',
    'dll': 'image/icon/type=DLL.svg',
    'dmg': 'image/icon/type=DMG.svg',
    'doc': 'image/icon/type=DOC.svg',
    'docx': 'image/icon/type=DOCX.svg',
    'dwg': 'image/icon/type=DWG.svg',
    'emf': 'image/icon/type=EMF.svg',
    'eps': 'image/icon/type=EPS.svg',
    'exe': 'image/icon/type=EXE.svg',
    'fig': 'image/icon/type=FIG.svg',
    'fla': 'image/icon/type=FLA.svg',
    'flac': 'image/icon/type=FLAC.svg',
    'flv': 'image/icon/type=FLV.svg',
    'fm': 'image/icon/type=FM.svg',
    'gif': 'image/icon/type=GIF.svg',
    'hlp': 'image/icon/type=HLP.svg',
    'html': 'image/icon/type=HTML.svg',
    'id': 'image/icon/type=ID.svg',
    'idml': 'image/icon/type=IDML.svg',
    'img': 'image/icon/type=IMG.svg',
    'indd': 'image/icon/type=INDD.svg',
    'inx': 'image/icon/type=INX.svg',
    'iso': 'image/icon/type=ISO.svg',
    'java': 'image/icon/type=JAVA.svg',
    'jpeg': 'image/icon/type=JPEG.svg',
    'jpg': 'image/icon/type=JPG.svg',
    'js': 'image/icon/type=JS.svg',
    'json': 'image/icon/type=JSON.svg',
    'm3u': 'image/icon/type=M3U.svg',
    'm4a': 'image/icon/type=M4A.svg',
    'mdb': 'image/icon/type=MDB.svg',
    'midi': 'image/icon/type=MIDI.svg',
    'mkv': 'image/icon/type=MKV.svg',
    'mov': 'image/icon/type=MOV.svg',
    'mp3': 'image/icon/type=MP3.svg',
    'mp4': 'image/icon/type=MP4.svg',
    'mpa': 'image/icon/type=MPA.svg',
    'mpeg': 'image/icon/type=MPEG.svg',
    'odt': 'image/icon/type=ODT.svg',
    'ogg': 'image/icon/type=OGG.svg',
    'otf': 'image/icon/type=OTF.svg',
    'pcm': 'image/icon/type=PCM.svg',
    'pdf': 'image/icon/type=PDF.svg',
    'php': 'image/icon/type=PHP.svg',
    'pkg': 'image/icon/type=PKG.svg',
    'pls': 'image/icon/type=PLS.svg',
    'png': 'image/icon/type=PNG.svg',
    'ppt': 'image/icon/type=PPT.svg',
    'pptx': 'image/icon/type=PPTX.svg',
    'ps': 'image/icon/type=PS.svg',
    'psd': 'image/icon/type=PSD.svg',
    'pub': 'image/icon/type=PUB.svg',
    'rar': 'image/icon/type=RAR.svg',
    'rav': 'image/icon/type=RAV.svg',
    'rss': 'image/icon/type=RSS.svg',
    'rtf': 'image/icon/type=RTF.svg',
    'sql': 'image/icon/type=SQL.svg',
    'svg': 'image/icon/type=SVG.svg',
    'swf': 'image/icon/type=SWF.svg',
    'tar': 'image/icon/type=TAR.svg',
    'tiff': 'image/icon/type=TIFF.svg',
    'ttf': 'image/icon/type=TTF.svg',
    'txt': 'image/icon/type=TXT.svg',
    'vob': 'image/icon/type=VOB.svg',
    'wav': 'image/icon/type=WAV.svg',
    'wma': 'image/icon/type=WMA.svg',
    'wmf': 'image/icon/type=WMF.svg',
    'wmv': 'image/icon/type=WMV.svg',
    'xd': 'image/icon/type=XD.svg',
    'xls': 'image/icon/type=XLS.svg',
    'xlsx': 'image/icon/type=XLSX.svg',
    'xml': 'image/icon/type=XML.svg',
    'zip': 'image/icon/type=ZIP.svg'
    };
    return iconMap[ext] || null;
}

// 渲染单个文件卡片
// 获取文件类型对应的 Emoji 图标（避免图片 404 闪烁）
function getFileIconEmoji(fileName) {
    if (!fileName) return '📁';
    const ext = fileName.split('.').pop().toLowerCase();
    const iconMap = {
        'pdf': '📄',
        'doc': '📝',
        'docx': '📝',
        'xls': '📊',
        'xlsx': '📊',
        'ppt': '📽️',
        'pptx': '📽️',
        'txt': '📋',
        'zip': '📦',
        'rar': '📦',
        '7z': '📦',
        'md': '📋',
        'exe': '⚙️',
        'py': '🐍',
        'js': '📜',
        'html': '🌐',
        'css': '🎨'
    };
    return iconMap[ext] || '📁';
}

function renderFileCard(fileData, currentCardIndex, fileIdx) {
    if (!fileData) return '';

    let filePath, suggestion, fileSize = '未知', modTime = '未知', bytes = 0;
    
    if (typeof fileData === 'object' && fileData !== null) {
        filePath = fileData.path || fileData.file_path || '';
        suggestion = fileData.suggestion || (fileIdx > 0 ? '删除' : '保留');
        
        // 健壮性：检查所有可能的体积字段
        bytes = fileData.size || fileData.file_size || fileData.file_size_bytes || 0;
        if (bytes) fileSize = formatFileSize(bytes);
        
        // 健壮性：检查所有可能的日期字段
        const mtime = fileData.mtime || fileData.mod_time || fileData.last_modified || fileData.modify_time;
        if (mtime) {
            const date = new Date(mtime * 1000);
            modTime = `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日`;
        }
    } else {
        filePath = String(fileData);
        suggestion = fileIdx > 0 ? '删除' : '保留';
    }
    
    const isChecked = suggestion.includes('删除');
    const fileName = filePath.split(/[\\/]/).pop() || '未命名';
    const displayFileName = formatFileName(fileName, 18);
    
    const displayPath = formatPath(filePath);
    const fileIcon = getFileIcon(fileName);
    const iconEmoji = getFileIconEmoji(fileName);

    return `
        <div class="file-card ${isChecked ? 'selected' : ''}" data-size="${bytes}" tabindex="0">
            <div class="file-card-row">
                <div class="custom-checkbox ${isChecked ? 'checked' : ''}" 
                     onclick="toggleCustomCheckbox(this)" 
                     data-path="${filePath.replace(/"/g, '&quot;')}">
                </div>
                <div class="file-card-content">
                    <div class="file-card-icon-wrapper">
                        ${fileIcon ? `<img src="${fileIcon}" class="file-card-icon">` : `<span class="file-icon-emoji">${iconEmoji}</span>`}
                    </div>
                    <div class="file-card-info">
                        <div class="file-card-name" title="${fileName}">${displayFileName}</div>
                        <div class="file-card-meta">
                            <span>${fileSize} | ${modTime}</span>
                        </div>
                        <div class="file-card-path" title="${filePath}">${displayPath}</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

let isDetailsExpanded = false;
function toggleDetails() {
    isDetailsExpanded = !isDetailsExpanded;
    const resultCard = document.getElementById('result-card');
    let detailsContainer = document.getElementById('details-expansion-container');
    
    if (!detailsContainer) {
        detailsContainer = document.createElement('div');
        detailsContainer.id = 'details-expansion-container';
        detailsContainer.className = 'glass-effect details-expansion';
        detailsContainer.style.display = 'none';
        resultCard.parentNode.insertBefore(detailsContainer, resultCard.nextSibling);
    }

    if (isDetailsExpanded) {
        const result = analysisResults[currentCardIndex];
        
        // 使用 LLM 返回的详细分析信息
        const llmDetail = result.analysis || '暂无详细分析信息';
        const similarity = result.similarity ? `${result.similarity}%` : '高';
        
        let detailsHtml = `
            <div class="details-content">
                <h4 class="details-title">AI 深度分析报告</h4>
                <div class="llm-reasoning glass-effect" style="padding: 12px; margin-bottom: 12px; font-size: 13px; line-height: 1.6; color: rgba(0,0,0,0.75); background: rgba(255,255,255,0.4);">
                    ${llmDetail}
                </div>
                <div class="details-grid">
                    <div class="details-item">
                        <span class="label">识别维度:</span>
                        <span class="value">${selectedMode === 'similar' ? '视觉/结构相似度' : '二进制数据一致性'}</span>
                    </div>
                    <div class="details-item">
                        <span class="label">置信度:</span>
                        <span class="value highlight">${similarity}</span>
                    </div>
                    <div class="details-item">
                        <span class="label">建议方案:</span>
                        <span class="value">AI 建议${result.type === 'similar' ? '保留清晰度更高或体积更小的版本' : '清理所有副本，仅保留一份原始文件'}</span>
                    </div>
                </div>
            </div>
        `;
        
        detailsContainer.innerHTML = detailsHtml;
        detailsContainer.style.display = 'block';
        // 触发重绘以应用动画
        detailsContainer.offsetHeight; 
        detailsContainer.style.animation = 'slideDown 0.3s ease-out forwards';
    } else {
        detailsContainer.style.animation = 'slideUp 0.3s ease-in forwards';
        setTimeout(() => { detailsContainer.style.display = 'none'; }, 300);
    }
}

function renderStackedCards() {
    const step4 = document.getElementById('step-4');
    const resultCard = document.getElementById('result-card');
    const backHomeBtn = document.getElementById('back-home-btn');
    const floatingBall = document.getElementById('floating-ball');
    
    // 关闭可能存在的详情页
    isDetailsExpanded = false;
    const oldDetails = document.getElementById('details-expansion-container');
    if (oldDetails) oldDetails.style.display = 'none';
    
    if (currentCardIndex >= analysisResults.length) {
        resultCard.innerHTML = `
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 300px; width: 100%; gap: 20px;">
                <p style="text-align: center; color: #64C998; padding: 20px 40px 0; font-size: 18px; font-weight: 600;">✅ 所有文件已处理完毕</p>
                <button class="btn-cleanup" style="width: 200px; background: linear-gradient(90deg, #64C998 0%, #52F7A5 100%);" onclick="goBackHome()">返回首页</button>
            </div>
        `;
        if (backHomeBtn) backHomeBtn.classList.add('hidden'); 
        
        // 恢复 logo
        if (floatingBall) {
            floatingBall.innerHTML = `<img src="image/logo.svg" alt="Logo" class="logo">`;
        }
        return;
    }
    
    if (backHomeBtn) backHomeBtn.classList.add('hidden');
    resultCard.dataset.index = currentCardIndex;
    resultCard.classList.remove('swipe-left', 'swipe-right');

    const result = analysisResults[currentCardIndex];
    const totalSizeMB = result.fileSize || 0;
    updateCarbonLevel(totalSizeMB);
    
    const modeLabels = { 
        'similar': '相似文件', 
        'duplicate': '重复文件', 
        'process': '过程文件',
        'history': '历史文件'
    };
    const cardTitle = modeLabels[selectedMode] || '发现文件';
    
    const sizeStr = result.fileSize ? `${result.fileSize.toFixed(1)}MB` : '0.0MB';
    const fileCount = (result.type === 'similar') ? 2 : (result.files ? result.files.length : 0);

    // 计算初始勾选数量
    let selectedCount = 0;
    if (result.type === 'similar') {
        selectedCount = 1; 
    } else if (result.files) {
        selectedCount = result.files.filter((f, i) => {
            if (typeof f === 'object' && f.suggestion) return f.suggestion.includes('删除');
            return i > 0;
        }).length;
    }

    // 1. 构建文件卡片的 HTML
    let filesToRender = [];
    if (result.type === 'similar') {
        filesToRender = [result.file1, result.file2];
    } else if (Array.isArray(result.files)) {
        filesToRender = result.files;
    }
    const renderedFilesHtml = filesToRender.map((file, idx) => renderFileCard(file, currentCardIndex, idx)).join('');

    // 2. 更新静态 HTML 中的内容
    const titleEl = document.getElementById('card-main-title');
    const sizeEl = document.getElementById('total-selected-size');
    const countEl = document.getElementById('selection-count');
    const fileListRow = document.getElementById('file-list-row');
    const reasonEl = document.getElementById('suggestion-reason');
    const cleanupBtn = document.getElementById('btn-cleanup');
    const ignoreBtn = document.getElementById('btn-ignore');

    if (titleEl) titleEl.textContent = cardTitle;
    if (sizeEl) sizeEl.textContent = sizeStr;
    if (countEl) countEl.textContent = `${selectedCount}/${fileCount} 个文件已选`;
    if (fileListRow) fileListRow.innerHTML = renderedFilesHtml;
    
    // 理由部分逻辑
    let reasonText = result.analysis || '这些文件内容相同，且在一段时间内未被使用';
    if (selectedMode === 'similar' || result.type === 'similar') {
        reasonText += '。点击“查看详细”后展开可以看见详细信息';
    }
    if (reasonEl) reasonEl.textContent = reasonText;
    
    // 更新按钮点击事件
    if (cleanupBtn) {
        cleanupBtn.disabled = false;
        cleanupBtn.textContent = '清理这组文件';
        cleanupBtn.onclick = () => handleCardConfirm(currentCardIndex);
    }
    if (ignoreBtn) {
        ignoreBtn.onclick = () => handleCardCancel(currentCardIndex);
    }

    // 更新 Logo 缩略图 (针对相似图片) - 用户要求不用更换，恢复原始 Logo
    if (floatingBall) {
        floatingBall.innerHTML = `<img src="image/logo.svg" alt="Logo" class="logo">`;
    }

    updateSelectionCount();
}

// Toggle custom checkbox
function toggleCustomCheckbox(el) {
    const isChecked = el.classList.toggle('checked');
    const card = el.closest('.file-card');
    if (card) {
        card.classList.toggle('selected', isChecked);
    }
    updateSelectionCount();
}

function updateSelectionCount() {
    const card = document.getElementById('result-card');
    if (!card) return;
    
    const checkboxes = card.querySelectorAll('.custom-checkbox');
    const checkedCheckboxes = card.querySelectorAll('.custom-checkbox.checked');
    const all = checkboxes.length;
    const checked = checkedCheckboxes.length;
    
    // 计算选中文件的总大小
    let totalBytes = 0;
    checkedCheckboxes.forEach(cb => {
        const fileCard = cb.closest('.file-card');
        if (fileCard) {
            totalBytes += parseInt(fileCard.dataset.size || 0);
        }
    });
    
    const sizeMB = (totalBytes / (1024 * 1024)).toFixed(1);
    
    // 更新左上方的大小显示
    const sizeEl = document.getElementById('total-selected-size');
    if (sizeEl) {
        sizeEl.textContent = `${sizeMB}MB`;
    }
    
    // 更新勾选数量显示
    const countEl = document.getElementById('selection-count');
    if (countEl) {
        countEl.textContent = `${checked}/${all} 个文件已选`;
    }
}

// 处理卡片取消操作
function handleCardCancel(index) {
    const card = document.getElementById('result-card');
    if (card && card.dataset.index == index) {
        card.classList.add('swipe-left');
        setTimeout(() => {
            currentCardIndex++;
            renderStackedCards();
        }, 300);
    }
}

// 处理卡片确认删除操作
async function handleCardConfirm(index) {
    const card = document.getElementById('result-card');
    if (card && card.dataset.index == index) {
        const confirmBtn = document.getElementById('btn-cleanup');
        if (confirmBtn) {
            confirmBtn.disabled = true;
            confirmBtn.textContent = '删除中...';
        }

        const checked = Array.from(card.querySelectorAll('.custom-checkbox.checked'));
        const paths = checked.map(cb => cb.getAttribute('data-path')).filter(Boolean);

        if (paths.length === 0) {
            if (confirmBtn) {
                confirmBtn.disabled = false;
                confirmBtn.textContent = '清理这组文件';
            }
            handleCardCancel(index);
            return;
        }

        if (!confirm(`确认将已勾选的 ${paths.length} 个文件移到回收站？`)) {
            if (confirmBtn) {
                confirmBtn.disabled = false;
                confirmBtn.textContent = '清理这组文件';
            }
            return;
        }

        try {
            const resp = await fetch('/api/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    root: lastAnalyzedPath,
                    paths: paths
                })
            });

            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || '删除失败');

            card.classList.add('swipe-right');
            setTimeout(() => {
                currentCardIndex++;
                renderStackedCards();
            }, 300);
        } catch (e) {
            console.error('删除请求失败:', e);
            alert(`删除失败: ${e.message}`);
            if (confirmBtn) {
                confirmBtn.disabled = false;
                confirmBtn.textContent = '清理这组文件';
            }
        }
    }
}

// 图片预览功能
function showImagePreview(imagePath) {
    const encodedPath = encodeURIComponent(imagePath);
    const imageUrl = `/api/image?path=${encodedPath}`;
    
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        animation: fadeIn 0.2s;
    `;
    
    const img = document.createElement('img');
    img.src = imageUrl;
    img.style.cssText = `
        max-width: 90%;
        max-height: 90%;
        border-radius: 8px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        cursor: default;
    `;
    
    overlay.onclick = (e) => {
        if (e.target === overlay) {
            overlay.style.animation = 'fadeOut 0.2s';
            setTimeout(() => overlay.remove(), 200);
        }
    };
    
    img.onclick = (e) => e.stopPropagation();
    
    const closeOnEsc = (e) => {
        if (e.key === 'Escape') {
            overlay.style.animation = 'fadeOut 0.2s';
            setTimeout(() => overlay.remove(), 200);
            document.removeEventListener('keydown', closeOnEsc);
        }
    };
    document.addEventListener('keydown', closeOnEsc);
    
    overlay.appendChild(img);
    document.body.appendChild(overlay);
}

function resetToStart() {
    currentStep = 1;
    showStep(1);
    analysisResults = [];
    currentCardIndex = 0;
    isAnalyzing = false;
    safeCloseEventSource();
    if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
    }
    resetProgress();
    document.getElementById('path-input').value = '';
    const backHomeBtn = document.getElementById('back-home-btn');
    if (backHomeBtn) backHomeBtn.classList.add('hidden');
}

function goBackHome() {
    resetToStart();
}

// ============ Step 2 - 文件夹选择相关函数 ============

let selectedFolders = []; // 存储已选择的文件夹对象

// #region agent log helper
function __agentLog(hypothesisId, location, message, data, runId = 'pre-fix') {
    try {
        fetch('http://127.0.0.1:7242/ingest/35825cd3-4cbb-4943-8cfe-85a066831bd9', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sessionId: 'debug-session',
                runId,
                hypothesisId,
                location,
                message,
                data,
                timestamp: Date.now()
            })
        }).catch(() => {});
    } catch (_) {}
}
// #endregion

// #region agent log (H6)
async function __selectFolderViaPywebview(trigger) {
    try {
        const api = window.pywebview && window.pywebview.api;
        const canPick = api && typeof api.select_folder === 'function';
        __agentLog(
            'H6',
            'document_sort/page/script.js:__selectFolderViaPywebview:entry',
            'attempt select_folder via pywebview',
            { trigger, canPick: !!canPick },
            'post-fix'
        );
        if (!canPick) return '';
        const chosen = await api.select_folder();
        const folderPath = (chosen || '').trim();
        __agentLog(
            'H6',
            'document_sort/page/script.js:__selectFolderViaPywebview:result',
            'select_folder result',
            { trigger, hasPath: !!folderPath, folderPath: folderPath ? folderPath.slice(0, 260) : null },
            'post-fix'
        );
        return folderPath;
    } catch (e) {
        __agentLog(
            'H6',
            'document_sort/page/script.js:__selectFolderViaPywebview:error',
            'select_folder threw error',
            { trigger, error: String(e && e.message ? e.message : e).slice(0, 260) },
            'post-fix'
        );
        return '';
    }
}
// #endregion

// 浏览文件夹（使用文件选择对话框）
function browseFolder(event) {
    if (event) event.preventDefault();
    
    console.log('点击浏览文件夹按钮');

    // 如果在 pywebview 中，优先用原生对话框拿到绝对路径（拖拽/浏览器无法保证拿到）
    if (window.pywebview && window.pywebview.api && typeof window.pywebview.api.select_folder === 'function') {
        __selectFolderViaPywebview('browseFolder').then((folderPath) => {
            if (folderPath) addFolderToList(folderPath);
        });
        return;
    }
    
    // 创建一个隐藏的input元素来触发文件选择
    const input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true; // Chrome/Edge
    input.directory = true;        // Firefox
    input.multiple = false;
    
    input.onchange = (e) => {
        console.log('文件选择对话框返回:', e.target.files);
        
        if (e.target.files && e.target.files.length > 0) {
            // 获取第一个文件的路径
            const firstFile = e.target.files[0];
            console.log('第一个文件:', firstFile);
            console.log('文件路径 (path):', firstFile.path);
            console.log('文件相对路径 (webkitRelativePath):', firstFile.webkitRelativePath);

            // #region agent log (H1/H2)
            __agentLog(
                'H1',
                'document_sort/page/script.js:browseFolder:onchange',
                'browseFolder onchange: firstFile fields',
                {
                    fileName: firstFile && firstFile.name,
                    size: firstFile && firstFile.size,
                    type: firstFile && firstFile.type,
                    hasPath: !!(firstFile && firstFile.path),
                    path: firstFile && firstFile.path ? String(firstFile.path).slice(0, 260) : null,
                    webkitRelativePath: firstFile && firstFile.webkitRelativePath ? String(firstFile.webkitRelativePath).slice(0, 260) : null,
                    userAgent: navigator.userAgent
                }
            );
            // #endregion
            
            // 尝试从 path 或 webkitRelativePath 提取文件夹路径
            let folderPath = '';
            
            if (firstFile.path) {
                // pywebview/Electron 环境：直接使用 path
                folderPath = firstFile.path;
                // 移除文件名，只保留文件夹路径
                const lastSep = Math.max(folderPath.lastIndexOf('/'), folderPath.lastIndexOf('\\'));
                if (lastSep > 0) {
                    folderPath = folderPath.substring(0, lastSep);
                }
                console.log('提取的文件夹路径:', folderPath);

                // #region agent log (H1)
                __agentLog(
                    'H1',
                    'document_sort/page/script.js:browseFolder:usePath',
                    'browseFolder extracted folderPath from firstFile.path',
                    { folderPath: String(folderPath).slice(0, 260) }
                );
                // #endregion

                addFolderToList(folderPath);
            } else if (firstFile.webkitRelativePath) {
                // 浏览器环境：webkitRelativePath 格式为 "文件夹名/文件名"
                const parts = firstFile.webkitRelativePath.split('/');
                if (parts.length > 0) {
                    folderPath = parts[0]; // 只有文件夹名，不是完整路径
                    console.log('提取的文件夹名:', folderPath);

                    // #region agent log (H2)
                    __agentLog(
                        'H2',
                        'document_sort/page/script.js:browseFolder:useRelative',
                        'browseFolder only has webkitRelativePath (no absolute path)',
                        { folderName: String(folderPath).slice(0, 260), webkitRelativePath: String(firstFile.webkitRelativePath).slice(0, 260) }
                    );
                    // #endregion

                    // 显示在输入框中，让用户补全路径
                    const pathInput = document.getElementById('folder-path-input');
                    if (pathInput) {
                        pathInput.value = folderPath;
                        pathInput.focus();
                        pathInput.select();
                        alert(`检测到文件夹：${folderPath}\n\n由于浏览器安全限制，请补全完整路径后按回车确认\n\n示例：\nWindows: D:\\Documents\\${folderPath}\nmacOS: /Users/用户名/Documents/${folderPath}`);
                    }
                }
            } else {
                console.warn('无法获取文件夹路径');

                // #region agent log (H1/H2)
                __agentLog(
                    'H3',
                    'document_sort/page/script.js:browseFolder:noPath',
                    'browseFolder: cannot get path or webkitRelativePath',
                    { keys: firstFile ? Object.keys(firstFile).slice(0, 50) : null }
                );
                // #endregion

                alert('无法获取文件夹路径\n请手动输入完整路径');
            }
        }
    };
    
    // 触发文件选择对话框
    input.click();
}

// 初始化文件夹选择界面
function initStep2() {
    const inputArea = document.getElementById('folder-input-area');
    const pathInput = document.getElementById('folder-path-input');
    
    if (!inputArea || !pathInput) return; // 元素可能尚未加载

    // #region agent log (H4)
    __agentLog(
        'H4',
        'document_sort/page/script.js:initStep2',
        'initStep2 bound',
        {
            hasInputArea: !!inputArea,
            hasPathInput: !!pathInput,
            userAgent: navigator.userAgent,
            isPywebview: !!window.pywebview
        }
    );
    // #endregion
    
    // 拖拽事件处理
    inputArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        inputArea.classList.add('drag-over');
    });
    
    inputArea.addEventListener('dragleave', () => {
        inputArea.classList.remove('drag-over');
    });
    
    inputArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        inputArea.classList.remove('drag-over');
        
        console.log('============ 拖入事件触发 ============');
        console.log('dataTransfer对象:', e.dataTransfer);

        // #region agent log (H1/H2/H5)
        __agentLog(
            'H5',
            'document_sort/page/script.js:drop:entry',
            'drop event triggered',
            {
                types: e.dataTransfer ? Array.from(e.dataTransfer.types || []) : null,
                filesLen: e.dataTransfer && e.dataTransfer.files ? e.dataTransfer.files.length : null,
                itemsLen: e.dataTransfer && e.dataTransfer.items ? e.dataTransfer.items.length : null,
                effectAllowed: e.dataTransfer ? e.dataTransfer.effectAllowed : null,
                dropEffect: e.dataTransfer ? e.dataTransfer.dropEffect : null,
                userAgent: navigator.userAgent,
                isPywebview: !!window.pywebview
            }
        );
        // #endregion
        
        // 尝试方法1: 使用 dataTransfer.files
        const files = e.dataTransfer.files;
        console.log('dataTransfer.files数量:', files.length);
        
        if (files && files.length > 0) {
            // #region agent log (H1/H2)
            const __first = files[0];
            __agentLog(
                'H1',
                'document_sort/page/script.js:drop:files[0]',
                'drop files[0] snapshot',
                {
                    name: __first && __first.name,
                    size: __first && __first.size,
                    type: __first && __first.type,
                    hasPath: !!(__first && __first.path),
                    path: __first && __first.path ? String(__first.path).slice(0, 260) : null,
                    webkitRelativePath: __first && __first.webkitRelativePath ? String(__first.webkitRelativePath).slice(0, 260) : null
                }
            );
            // #endregion

            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                console.log(`\n文件 ${i}:`, file);
                console.log('- name:', file.name);
                console.log('- size:', file.size);
                console.log('- type:', file.type);
                console.log('- path:', file.path);
                console.log('- webkitRelativePath:', file.webkitRelativePath);
                console.log('- lastModified:', file.lastModified);
                
                // 尝试获取所有可能的属性
                console.log('- 所有属性:', Object.keys(file));
                console.log('- 所有属性值:');
                for (let key in file) {
                    if (typeof file[key] !== 'function') {
                        console.log(`  ${key}:`, file[key]);
                    }
                }
                
                // 方法1a: 使用 file.path (pywebview/Electron)
                if (file.path) {
                    console.log('✅ 成功获取路径 (file.path):', file.path);

                    // #region agent log (H1)
                    __agentLog(
                        'H1',
                        'document_sort/page/script.js:drop:useFilePath',
                        'drop: got file.path (treating as folder path)',
                        { filePath: String(file.path).slice(0, 260) }
                    );
                    // #endregion

                    addFolderToList(file.path);
                    return;
                }
                
                // 方法1b: 如果size为0且没有type，可能是文件夹
                if (file.size === 0 && !file.type && file.name) {
                    console.log('⚠️ 检测到可能的空文件夹（size=0, no type）:', file.name);

                    // #region agent log (H2)
                    __agentLog(
                        'H2',
                        'document_sort/page/script.js:drop:emptyFolderHeuristic',
                        'drop: size=0 & no type heuristic triggered',
                        { name: String(file.name).slice(0, 260) }
                    );
                    // #endregion

                    pathInput.value = file.name;
                    pathInput.focus();
                    pathInput.select();
                    alert(`检测到文件夹：${file.name}\n\n请在输入框中补全完整路径后按回车确认\n例如：D:\\Documents\\${file.name}`);
                    return;
                }
            }
        }
        
        // 尝试方法2: 使用 DataTransferItemList API
        const items = e.dataTransfer.items;
        console.log('\n尝试 DataTransferItemList API');
        console.log('dataTransfer.items数量:', items ? items.length : 0);
        
        if (items && items.length > 0) {
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                console.log(`\nItem ${i}:`, item);
                console.log('- kind:', item.kind);
                console.log('- type:', item.type);
                
                if (item.kind === 'file') {
                    // 尝试获取File对象
                    const file = item.getAsFile();
                    console.log('- getAsFile():', file);
                    if (file && file.path) {
                        console.log('✅ 成功获取路径 (item.getAsFile().path):', file.path);
                        addFolderToList(file.path);
                        return;
                    }
                    
                    // 尝试获取Entry对象
                    const entry = item.webkitGetAsEntry();
                    console.log('- webkitGetAsEntry():', entry);
                    
                    if (entry) {
                        console.log('  - name:', entry.name);
                        console.log('  - fullPath:', entry.fullPath);
                        console.log('  - isDirectory:', entry.isDirectory);
                        console.log('  - isFile:', entry.isFile);

                        // #region agent log (H2)
                        __agentLog(
                            'H2',
                            'document_sort/page/script.js:drop:entry',
                            'drop: webkitGetAsEntry snapshot',
                            {
                                name: entry.name,
                                fullPath: entry.fullPath,
                                isDirectory: !!entry.isDirectory,
                                isFile: !!entry.isFile
                            }
                        );
                        // #endregion
                        
                        if (entry.isDirectory) {
                            console.log('⚠️ 检测到文件夹（但无法获取完整路径）:', entry.name);
                            // 在 pywebview 中：拖拽也可能拿不到绝对路径，直接弹出原生选择框兜底
                            const picked = await __selectFolderViaPywebview('drop:directory');
                            if (picked) {
                                addFolderToList(picked);
                                return;
                            }

                            // 浏览器/无法调用 pywebview API：只能让用户手动补全
                            pathInput.value = entry.name;
                            pathInput.focus();
                            pathInput.select();
                            alert(`检测到文件夹：${entry.name}\n\n当前环境无法从拖拽中获取磁盘绝对路径（这是浏览器/WebView 的安全限制）\n你可以：\n1) 在输入框中补全完整路径后按回车确认\n2) 或点击该区域弹出“选择文件夹”对话框\n\n示例：\nWindows: D:\\Documents\\${entry.name}\nmacOS: /Users/用户名/Documents/${entry.name}`);
                            return;
                        }
                    }
                }
            }
        }
        
        // 尝试方法3: 检查 dataTransfer.types
        console.log('\ndataTransfer.types:', e.dataTransfer.types);
        console.log('dataTransfer.effectAllowed:', e.dataTransfer.effectAllowed);
        console.log('dataTransfer.dropEffect:', e.dataTransfer.dropEffect);
        
        // 如果所有方法都失败了
        console.log('\n❌ 所有方法都无法获取文件夹路径');
        console.log('可能的原因:');
        console.log('1. pywebview 不支持拖拽获取路径');
        console.log('2. 浏览器安全限制');
        console.log('3. 操作系统限制');
        alert('无法自动获取文件夹路径\n\n请手动输入完整路径：\n\nWindows示例：D:\\Documents\\MyFolder\nmacOS示例：/Users/用户名/Documents/MyFolder');
        pathInput.focus();
    });

    // 点击区域：尽量弹出原生选择框（pywebview），否则聚焦输入框
    inputArea.addEventListener('click', async () => {
        const picked = await __selectFolderViaPywebview('click:inputArea');
        if (picked) {
            addFolderToList(picked);
            return;
        }
        pathInput.focus();
    });
    
    // 输入框回车确认
    pathInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const folderPath = pathInput.value.trim();
            if (folderPath) {
                addFolderToList(folderPath);
                pathInput.value = '';
            }
        }
    });
}

// ============ Step 3 - 进度/日志 hover 交互（不影响日志滚动逻辑） ============
function initStep3() {
    const step3 = document.getElementById('step-3');
    const logContainer = document.getElementById('log-container');
    const progressWrapper = document.getElementById('progress-wrapper');
    if (!step3 || !logContainer || !progressWrapper) return;

    const show = () => step3.classList.add('show-progress-on-hover');
    const hide = () => step3.classList.remove('show-progress-on-hover');

    logContainer.addEventListener('mouseenter', show);
    logContainer.addEventListener('mouseleave', hide);
}

// 添加文件夹到列表
async function addFolderToList(folderPath) {
    // 去除路径两端的空格和引号
    folderPath = folderPath.trim().replace(/^["']|["']$/g, '');
    
    // 检查是否已存在
    if (selectedFolders.some(f => f.path === folderPath)) {
        alert('该文件夹已添加');
        return;
    }
    
    // 创建文件夹对象
    const folderObj = {
        path: folderPath,
        name: folderPath.split(/[\/\\]/).pop() || folderPath,
        size: '计算中...',
        checked: true  // 默认选中
    };
    
    console.log('创建的文件夹对象:', folderObj);
    
    // 使用 unshift 将新文件夹添加到列表开头（最前面）
    selectedFolders.unshift(folderObj);
    console.log('添加后的文件夹列表:', selectedFolders);
    
    renderFoldersList();
    updateConfirmButton();
    
    // 异步获取文件夹大小
    try {
        const response = await fetch('/api/get_folder_size', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: folderPath })
        });
        
        if (response.ok) {
            const data = await response.json();
            // 更新文件夹大小
            const folder = selectedFolders.find(f => f.path === folderPath);
            if (folder) {
                const sizeInMB = (data.size / (1024 * 1024)).toFixed(1);
                folder.size = `${sizeInMB}MB`;
                renderFoldersList();
            }
        } else {
            // 如果后端不支持，显示未知
            const folder = selectedFolders.find(f => f.path === folderPath);
            if (folder) {
                folder.size = '未知';
                renderFoldersList();
            }
        }
    } catch (error) {
        console.log('无法获取文件夹大小:', error);
        // 出错时显示未知
        const folder = selectedFolders.find(f => f.path === folderPath);
        if (folder) {
            folder.size = '未知';
            renderFoldersList();
        }
    }
}

// 渲染文件夹列表
function renderFoldersList() {
    const foldersList = document.getElementById('folders-list');
    console.log('渲染文件夹列表，元素:', foldersList);
    console.log('当前文件夹数量:', selectedFolders.length);
    console.log('文件夹详情:', selectedFolders);
    
    if (!foldersList) {
        console.error('未找到 folders-list 元素');
        return;
    }
    
    if (selectedFolders.length === 0) {
        // 显示空状态（根据Figma设计）
        foldersList.innerHTML = `
            <div class="folders-list-empty">
                <div class="empty-icon-wrapper">
                    <img class="empty-folder-icon" src="image/empty_folder.svg" alt="Empty Folder" />
                </div>
                <p class="empty-text">这里暂时还没有文件夹</p>
            </div>
        `;
        return;
    }
    
    foldersList.innerHTML = selectedFolders.map((folder, index) => `
        <div class="folder-item" data-index="${index}">
            <div class="folder-item-row">
                <div class="folder-checkbox-wrapper">
                    <div class="folder-checkbox ${folder.checked ? 'checked' : ''}" 
                         onclick="toggleFolderCheck(${index})"></div>
                </div>
                <div class="folder-info-row">
                    <div class="folder-icon-wrapper">
                        <img class="folder-icon" src="image/folder.svg" alt="Folder Icon" width="32" height="32"/ >
                        </svg>
                    </div>
                    <div class="folder-details">
                        <span class="folder-name" title="${folder.path}">${folder.name}</span>
                        <div class="folder-size-wrapper">
                            <span class="folder-size">${folder.size}</span>
                        </div>
                    </div>
                </div>
                <div class="delete-icon" onclick="deleteFolderFromList(${index})"></div>
            </div>
        </div>
    `).join('');
}

// 切换文件夹选中状态
function toggleFolderCheck(index) {
    if (selectedFolders[index]) {
        selectedFolders[index].checked = !selectedFolders[index].checked;
        renderFoldersList();
        updateConfirmButton();
    }
}

// 从列表中删除文件夹
function deleteFolderFromList(index) {
    console.log('删除文件夹，索引:', index);
    console.log('删除前文件夹列表:', selectedFolders);
    selectedFolders.splice(index, 1);
    console.log('删除后文件夹列表:', selectedFolders);
    renderFoldersList();
    updateConfirmButton();
}

// 更新确认按钮状态
function updateConfirmButton() {
    const confirmBtn = document.getElementById('btn-step2-confirm');
    if (!confirmBtn) return;
    
    // 检查是否有选中的文件夹
    const hasChecked = selectedFolders.some(f => f.checked);
    confirmBtn.disabled = !hasChecked;
}

// 确认文件夹选择
function confirmFolderSelection() {
    const checkedFolders = selectedFolders.filter(f => f.checked);
    
    if (checkedFolders.length === 0) {
        alert('请至少选择一个文件夹');
        return;
    }
    
    // 如果只选择了一个文件夹，直接使用该路径
    if (checkedFolders.length === 1) {
        const folderPath = checkedFolders[0].path;
        console.log('开始分析文件夹:', folderPath);
        goToStep3(folderPath);
    } else {
        // 多个文件夹：可以传递多个路径或合并处理（需要后端支持）
        // 暂时只分析第一个
        alert('当前仅支持单文件夹分析，将分析第一个选中的文件夹');
        const folderPath = checkedFolders[0].path;
        console.log('开始分析文件夹:', folderPath);
        goToStep3(folderPath);
    }
}

// 取消文件夹选择
function cancelFolderSelection() {
    selectedFolders = [];
    renderFoldersList();
    updateConfirmButton();
    goBackToStep1();
}

// 页面加载时初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initStep2();
        initStep3();
        renderFoldersList();  // 初始化时渲染空状态
        updateConfirmButton();
    });
} else {
    initStep2();
    initStep3();
    renderFoldersList();  // 初始化时渲染空状态
    updateConfirmButton();
}
