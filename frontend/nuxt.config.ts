// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },
  modules: ['@nuxtjs/tailwindcss'],
  runtimeConfig: {
    public: {
      apiUrl: process.env.API_URL || 'http://localhost:8000'
    }
  },
  app: {
    head: {
      title: 'Mandi Intelligence – BeejRakshak',
      meta: [
        { name: 'description', content: 'ML-powered price predictions for Gujarat farmers.' }
      ],
      link: [
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap' }
      ]
    }
  }
})
