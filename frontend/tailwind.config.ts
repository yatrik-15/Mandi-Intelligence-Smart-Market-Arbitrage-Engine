import type { Config } from 'tailwindcss'

export default {
    content: [
        './app/components/**/*.{js,vue,ts}',
        './app/layouts/**/*.vue',
        './app/pages/**/*.vue',
        './app/plugins/**/*.{js,ts}',
        './app/app.vue',
        './app/error.vue'
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
            colors: {
                brand: {
                    beige: '#e8e2d8',
                    bg: '#fdfbf7', // Lighter off-white for the main background
                    card: '#ffffff', // White cards
                    card2: '#f5f0e6', // Slight golden hay tint for secondary elements
                    border: '#dcd1c1', // Tan border
                    muted: '#8b7d6b' // Warm brownish muted text
                }
            }
        }
    },
    plugins: [],
} satisfies Config
