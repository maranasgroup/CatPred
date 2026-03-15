<template>
  <footer class="footer">
    <div class="container footer-inner">
      <p class="footer-text">
        <span class="footer-brand">CatPred</span>
        <span class="footer-sep">&middot;</span>
        Maranas Group
        <span class="footer-sep">&middot;</span>
        Penn State University
        <span class="footer-sep">&middot;</span>
        2026
      </p>
    </div>
  </footer>
</template>

<style scoped>
.footer {
  border-top: 1px solid var(--border);
  padding: 1.5rem 0;
}

.footer-inner {
  text-align: center;
}

.footer-text {
  color: var(--text-tertiary);
  font-size: 0.8rem;
}

.footer-brand {
  font-family: var(--font-serif);
  color: var(--text-secondary);
}

.footer-sep {
  margin: 0 0.25rem;
  opacity: 0.5;
}
</style>
