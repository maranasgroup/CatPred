<template>
  <a href="#main-content" class="skip-link">Skip to content</a>
</template>

<style scoped>
.skip-link {
  position: absolute;
  top: -100%;
  left: 1rem;
  z-index: 999;
  padding: 0.5rem 1rem;
  background: var(--accent);
  color: #fff;
  border-radius: var(--radius-sm);
  font-size: 0.875rem;
  font-weight: 500;
}

.skip-link:focus {
  top: 0.5rem;
}
</style>
