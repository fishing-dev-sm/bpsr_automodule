// 模组OCR优化器 Web UI JavaScript

class ModuleOptimizerApp {
    constructor() {
        this.isCalculating = false;
        this.currentResults = null;
        this.initializeApp();
    }

    initializeApp() {
        this.bindEvents();
        this.checkScreenshotStatus();
    }

    bindEvents() {
        // 开始计算按钮
        document.getElementById('start-btn').addEventListener('click', () => {
            this.startCalculation();
        });

        // 分组筛选器
        document.getElementById('group-filter').addEventListener('change', (e) => {
            this.filterCombinations(e.target.value);
        });
    }

    async checkScreenshotStatus() {
        try {
            const response = await fetch('/api/screenshot_info');
            const data = await response.json();
            
            if (response.ok) {
                const statusEl = document.getElementById('file-status');
                const startBtn = document.getElementById('start-btn');
                
                statusEl.textContent = `就绪 (${data.total_files} 张截图)`;
                statusEl.style.color = '#059669';
                startBtn.disabled = false;
            } else {
                this.showError('截图文件夹检查失败: ' + data.error);
            }
        } catch (error) {
            this.showError('无法连接到服务器: ' + error.message);
        }
    }

    async startCalculation() {
        if (this.isCalculating) return;

        try {
            const response = await fetch('/api/start_calculation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                this.isCalculating = true;
                this.showProgress();
                this.hideError();
                this.hideResults();
                this.pollCalculationStatus();
            } else {
                const data = await response.json();
                this.showError(data.error || '启动计算失败');
            }
        } catch (error) {
            this.showError('网络错误: ' + error.message);
        }
    }

    showProgress() {
        const progressSection = document.getElementById('progress-section');
        const startBtn = document.getElementById('start-btn');
        
        progressSection.style.display = 'block';
        startBtn.disabled = true;
        startBtn.innerHTML = '<div class="loading"></div> 计算中...';
    }

    hideProgress() {
        const progressSection = document.getElementById('progress-section');
        const startBtn = document.getElementById('start-btn');
        
        progressSection.style.display = 'none';
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-rocket"></i> 开始OCR识别和优化计算';
    }

    async pollCalculationStatus() {
        const pollInterval = 1000; // 1秒轮询一次

        const poll = async () => {
            try {
                const response = await fetch('/api/calculation_status');
                const status = await response.json();

                this.updateProgress(status);

                if (status.completed) {
                    this.isCalculating = false;
                    this.hideProgress();
                    await this.loadResults();
                } else if (status.error) {
                    this.isCalculating = false;
                    this.hideProgress();
                    this.showError('计算错误: ' + status.error);
                } else if (status.running) {
                    setTimeout(poll, pollInterval);
                }
            } catch (error) {
                this.isCalculating = false;
                this.hideProgress();
                this.showError('状态查询失败: ' + error.message);
            }
        };

        poll();
    }

    updateProgress(status) {
        const progressText = document.getElementById('progress-text');
        const progressPercent = document.getElementById('progress-percent');
        const progressFill = document.getElementById('progress-fill');

        progressText.textContent = status.current_step || '处理中...';
        progressPercent.textContent = `${status.progress}%`;
        progressFill.style.width = `${status.progress}%`;
    }

    async loadResults() {
        try {
            const response = await fetch('/api/results');
            const data = await response.json();

            if (response.ok) {
                this.currentResults = data;
                this.displayResults(data);
            } else {
                this.showError('获取结果失败: ' + data.error);
            }
        } catch (error) {
            this.showError('网络错误: ' + error.message);
        }
    }

    displayResults(data) {
        // 显示结果区域
        document.getElementById('results-section').style.display = 'block';

        // 显示概要信息
        this.displaySummary(data.summary);

        // 显示识别的模组
        this.displayModules(data.modules);

        // 显示最优组合
        this.displayCombinations(data.grouped_combinations);

        // 设置分组筛选器
        this.setupGroupFilter(data.grouped_combinations);
    }

    displaySummary(summary) {
        document.getElementById('total-modules').textContent = summary.total_modules;
        document.getElementById('total-combinations').textContent = summary.total_combinations;
        document.getElementById('max-score').textContent = summary.max_score;
        document.getElementById('calc-time').textContent = summary.calculation_time || '计算中...';
    }

    displayModules(modules) {
        const container = document.getElementById('modules-grid');
        container.innerHTML = '';

        for (const [moduleName, moduleData] of Object.entries(modules)) {
            const moduleCard = this.createModuleCard(moduleName, moduleData);
            container.appendChild(moduleCard);
        }
    }

    createModuleCard(name, moduleData) {
        const card = document.createElement('div');
        card.className = 'module-card';
        
        // 添加品质样式类
        const qualityClass = this.getQualityClass(moduleData.quality);
        card.classList.add(qualityClass);

        let attributesHtml = Object.entries(moduleData.attributes)
            .map(([attrName, value]) => `
                <div class="attribute-item">
                    <span class="attribute-name">${attrName}</span>
                    <span class="attribute-value">${value}</span>
                </div>
            `).join('');

        // 若识别的词条不足，使用占位词条补齐显示数量（仅显示“+数字”占位，不映射属性名）
        const displayedCount = Object.keys(moduleData.attributes).length;
        const targetCount = moduleData.inferred_entry_count || moduleData.attribute_count || displayedCount;
        const placeholders = moduleData.placeholder_values || [];
        if (displayedCount < targetCount && placeholders.length > 0) {
            const additional = placeholders.slice(0, Math.max(0, targetCount - displayedCount))
                .map(v => `
                    <div class="attribute-item">
                        <span class="attribute-name">(未识别)</span>
                        <span class="attribute-value">${v}</span>
                    </div>
                `).join('');
            attributesHtml += additional;
        }

        const entryCount = moduleData.attribute_count || Object.keys(moduleData.attributes).length;
        const inferredCount = moduleData.inferred_entry_count || entryCount;

        card.innerHTML = `
            <div class="module-header">
                <div class="module-name">${name}</div>
                <div class="module-quality ${qualityClass}">${moduleData.quality_name}（${inferredCount}条）</div>
            </div>
            <div class="module-attributes">
                ${attributesHtml}
            </div>
        `;

        return card;
    }

    getQualityClass(quality) {
        const qualityMap = {
            'legendary': 'quality-legendary',
            'epic': 'quality-epic',
            'rare': 'quality-rare',
            'common': 'quality-common',
            'unknown': 'quality-unknown'
        };
        return qualityMap[quality] || 'quality-unknown';
    }

    displayCombinations(groupedCombinations) {
        const container = document.getElementById('combinations-container');
        container.innerHTML = '';

        for (const [groupName, combinations] of Object.entries(groupedCombinations)) {
            const groupElement = this.createCombinationGroup(groupName, combinations);
            container.appendChild(groupElement);
        }
    }

    createCombinationGroup(groupName, combinations) {
        const group = document.createElement('div');
        group.className = 'combination-group';
        group.dataset.group = groupName;

        const combinationsHtml = combinations
            .map((combo, index) => this.createCombinationItemHtml(combo, index + 1))
            .join('');

        group.innerHTML = `
            <div class="group-header">
                <i class="fas fa-star"></i> ${groupName} 最大化组合 (${combinations.length}个方案)
            </div>
            <div class="group-content">
                ${combinationsHtml}
            </div>
        `;

        return group;
    }

    createCombinationItemHtml(combo, index) {
        // 生成模组详细信息HTML
        const moduleDetailsHtml = (combo.module_details || [])
            .map(module => {
                let attrsHtml = Object.entries(module.attributes)
                    .map(([attrName, value]) => `${attrName}+${value}`)
                    .join(', ');
                // 组合中同样补齐未识别词条的占位
                const displayed = Object.keys(module.attributes).length;
                const target = module.inferred_entry_count || module.attribute_count || displayed;
                const ph = module.placeholder_values || [];
                if (displayed < target && ph.length > 0) {
                    const extra = ph.slice(0, Math.max(0, target - displayed))
                        .map(v => `(+${v})`)
                        .join(', ');
                    attrsHtml = attrsHtml ? `${attrsHtml}, ${extra}` : extra;
                }
                
                const qualityClass = this.getQualityClass(module.quality);
                
                return `
                    <div class="module-detail">
                        <div class="module-detail-header">
                            <div class="module-detail-name">${module.name}</div>
                            <div class="module-detail-quality ${qualityClass}">${module.quality_name}（${module.inferred_entry_count || module.attribute_count || Object.keys(module.attributes).length}条）</div>
                        </div>
                        <div class="module-detail-attrs">${attrsHtml || '无属性数据'}</div>
                    </div>
                `;
            })
            .join('');

        const attributesHtml = Object.entries(combo.attributes)
            .map(([attrName, attrData]) => `
                <div class="attribute-badge ${attrData.is_maxed ? 'maxed' : ''}">
                    <span class="attribute-name">${attrName}</span>
                    <div>
                        <span class="attribute-value">${attrData.value}</span>
                        <span class="attribute-efficiency">(${attrData.efficiency})</span>
                    </div>
                </div>
            `).join('');

        return `
            <div class="combination-item">
                <div class="combination-header">
                    <div class="combination-title">方案 ${index}</div>
                    <div class="combination-score">${combo.maxed_count || 0}个属性达到20+</div>
                </div>
                <div class="combination-modules">
                    <strong>模组组合 (${combo.module_count}/4):</strong>
                    <div class="modules-details">
                        ${moduleDetailsHtml}
                    </div>
                </div>
                <div class="combination-attributes">
                    <div class="attributes-header"><strong>总计属性:</strong></div>
                    <div class="attributes-content">
                        ${attributesHtml}
                    </div>
                </div>
            </div>
        `;
    }

    setupGroupFilter(groupedCombinations) {
        const filterSelect = document.getElementById('group-filter');
        filterSelect.innerHTML = '<option value="">所有组合</option>';

        for (const groupName of Object.keys(groupedCombinations)) {
            const option = document.createElement('option');
            option.value = groupName;
            option.textContent = groupName;
            filterSelect.appendChild(option);
        }
    }

    filterCombinations(selectedGroup) {
        const allGroups = document.querySelectorAll('.combination-group');
        
        allGroups.forEach(group => {
            if (!selectedGroup || group.dataset.group === selectedGroup) {
                group.style.display = 'block';
            } else {
                group.style.display = 'none';
            }
        });
    }

    showError(message) {
        const errorEl = document.getElementById('error-message');
        const errorText = document.getElementById('error-text');
        
        errorText.textContent = message;
        errorEl.style.display = 'flex';
        
        // 3秒后自动隐藏错误消息
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        document.getElementById('error-message').style.display = 'none';
    }

    showResults() {
        document.getElementById('results-section').style.display = 'block';
    }

    hideResults() {
        document.getElementById('results-section').style.display = 'none';
    }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new ModuleOptimizerApp();
});
