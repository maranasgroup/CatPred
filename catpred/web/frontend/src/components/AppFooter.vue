<template>
  <footer class="footer">
    <div class="container footer-inner">
      <p class="footer-text">
        <span class="footer-brand">CatPred</span>
        <span class="footer-sep">&middot;</span>
        Developed in
        <a href="https://www.maranasgroup.com/" target="_blank" rel="noreferrer" class="footer-link">Maranas Group</a>
        at Penn State
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

.footer-link {
  color: var(--text-secondary);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}

.footer-link:hover {
  color: var(--accent);
  border-bottom-color: var(--accent);
}
</style>
