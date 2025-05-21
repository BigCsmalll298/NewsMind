class TypeWriter {
    constructor(element, texts = [], options = {}) {
        this.element = element;
        this.texts = texts;
        this.currentTextIndex = 0;
        this.currentCharIndex = 0;
        this.isDeleting = false;
        this.options = {
            typeSpeed: options.typeSpeed || 100,
            deleteSpeed: options.deleteSpeed || 50,
            delayBeforeDelete: options.delayBeforeDelete || 2000,
            delayBeforeNext: options.delayBeforeNext || 500
        };
    }

    type() {
        const currentText = this.texts[this.currentTextIndex];
        
        if (this.isDeleting) {
            // 删除文字
            this.element.innerHTML = currentText.substring(0, this.currentCharIndex - 1);
            this.currentCharIndex--;
            
            if (this.currentCharIndex === 0) {
                this.isDeleting = false;
                this.currentTextIndex = (this.currentTextIndex + 1) % this.texts.length;
                setTimeout(() => this.type(), this.options.delayBeforeNext);
                return;
            }
            
            setTimeout(() => this.type(), this.options.deleteSpeed);
        } else {
            // 添加文字
            this.element.innerHTML = currentText.substring(0, this.currentCharIndex + 1);
            this.currentCharIndex++;
            
            if (this.currentCharIndex === currentText.length) {
                this.isDeleting = true;
                setTimeout(() => this.type(), this.options.delayBeforeDelete);
                return;
            }
            
            setTimeout(() => this.type(), this.options.typeSpeed);
        }
    }

    start() {
        if (this.texts.length === 0) return;
        this.type();
    }
}

