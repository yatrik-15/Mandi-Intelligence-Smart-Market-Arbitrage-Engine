<template>
  <div class="min-h-screen text-slate-800 flex flex-col" style="background-color: #e8e2d8">
    <!-- Sticky Navbar -->
    <header class="w-full py-3 px-6 md:px-10 flex items-center gap-3 backdrop-blur-xl sticky top-0 z-[100] border-b border-[#d5ccbe]/60" style="background-color: rgba(232, 226, 216, 0.85)">
      <div class="flex items-center gap-2.5">
        <div class="w-9 h-9 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center text-white text-lg shadow-md shadow-green-600/20">🌾</div>
        <div>
          <h1 class="text-base font-extrabold text-slate-900 leading-tight tracking-tight">BeejRakshak <span class="text-green-600">Mandi</span></h1>
          <p class="text-[10px] text-[#8b7d6b] font-semibold tracking-wider uppercase leading-none">Smart Market Intelligence</p>
        </div>
      </div>
      <nav class="ml-auto hidden md:flex items-center gap-6">
        <a href="#video-section" class="text-xs font-bold text-slate-600 hover:text-green-600 transition-colors tracking-wide uppercase">How it Works</a>
        <a href="#features-section" class="text-xs font-bold text-slate-600 hover:text-green-600 transition-colors tracking-wide uppercase">Features</a>
        <a href="#form-section" class="text-xs font-bold text-slate-600 hover:text-green-600 transition-colors tracking-wide uppercase">Try Now</a>
      </nav>
      <div class="ml-auto md:ml-4 bg-green-600/10 text-green-700 border border-green-600/25 rounded-full px-3 py-1 text-[10px] font-black tracking-[0.15em] uppercase">
        AI Powered
      </div>
    </header>

    <NuxtPage />

    <!-- Footer -->
    <footer class="w-full border-t border-[#d5ccbe]/60 py-10 px-6 md:px-10">
      <div class="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div class="flex items-center gap-2">
          <span class="text-lg">🌾</span>
          <span class="text-sm font-bold text-slate-700">BeejRakshak</span>
          <span class="text-xs text-[#8b7d6b]">© 2026</span>
        </div>
        <p class="text-xs text-[#8b7d6b] text-center md:text-right leading-relaxed">
          ML-powered price intelligence for Indian farmers. Built with Prophet, TFT & CatBoost.<br class="hidden md:block"/>
          Data sourced from AGMARKNET. Made in Gujarat 🇮🇳
        </p>
      </div>
    </footer>
  </div>
</template>
