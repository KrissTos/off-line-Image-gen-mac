/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:      '#0a0a0a',
        surface: '#141414',
        card:    '#1c1c1c',
        border:  '#2a2a2a',
        accent:  '#7c3aed',
        muted:   '#6b7280',
      },
    },
  },
  plugins: [require('@tailwindcss/forms')],
}

