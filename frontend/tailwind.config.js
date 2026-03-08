/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:      'rgb(var(--color-bg) / <alpha-value>)',
        surface: 'rgb(var(--color-surface) / <alpha-value>)',
        card:    'rgb(var(--color-card) / <alpha-value>)',
        border:  'rgb(var(--color-border) / <alpha-value>)',
        accent:  'rgb(var(--color-accent) / <alpha-value>)',
        muted:   'rgb(var(--color-muted) / <alpha-value>)',
        label:   'rgb(var(--color-label) / <alpha-value>)',
      },
    },
  },
  plugins: [require('@tailwindcss/forms')],
}
