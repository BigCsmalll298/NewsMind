// Live2D 初始化配置
window.addEventListener('DOMContentLoaded', () => {
    // 构建模型的完整路径
    const basePath = window.APPLICATION_ROOT || '';
    const modelPath = `${basePath}/static/resources/model/hijiki/hijiki.model.json`;
    
    // 加载Live2D模型
    loadlive2d("live2d", modelPath);
}); 