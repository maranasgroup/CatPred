<template>
  <header class="header">
    <div class="container header-inner">
      <a href="/" class="brand" aria-label="CatPred home">
        <svg class="brand-icon" viewBox="0 0 32 32" width="22" height="22" fill="none" aria-hidden="true">
          <path d="M6 24C8 16 11 9 16 6C21 9 24 16 26 24" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" opacity="0.7" />
          <path d="M10 23C12 17 14 12 16 10C18 12 20 17 22 23" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" opacity="0.45" />
        </svg>
        <span class="brand-name">CatPred</span>
      </a>
      <nav class="header-nav" aria-label="Main navigation">
        <a
          href="https://www.nature.com/articles/s41467-025-57215-9"
          target="_blank"
          rel="noreferrer"
          class="nav-link"
        >Nat. Commun.</a>
        <a
          href="https://github.com/maranasgroup/catpred/"
          target="_blank"
          rel="noreferrer"
          class="nav-link"
        >GitHub</a>
      </nav>
    </div>
  </header>
</template>

<style scoped>
.header {
  position: sticky;
  top: 0;
  z-index: 50;
  background: rgba(250, 250, 249, 0.85);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
}

.header-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 56px;
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.brand-icon {
  color: var(--accent);
}

.brand-name {
  font-family: var(--font-serif);
  font-size: 1.15rem;
  font-weight: 400;
  letter-spacing: -0.02em;
  color: var(--text);
}

.header-nav {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.nav-link {
  display: inline-flex;
  align-items: center;
  height: 32px;
  padding: 0 0.75rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: var(--text-secondary);
  transition: color 0.15s, border-color 0.15s, background 0.15s;
}

.nav-link:hover {
  color: var(--text);
  border-color: var(--accent);
  background: var(--accent-light);
}
</style>
