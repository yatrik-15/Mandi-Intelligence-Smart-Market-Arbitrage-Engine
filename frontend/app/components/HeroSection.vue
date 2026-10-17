<template>
  <section class="w-full flex flex-col items-center">

    <!-- Compact Headline (above video, not competing with it) -->
    <div class="w-full relative z-10 pt-12 pb-6 px-4 flex flex-col items-center">
      <div class="max-w-3xl text-center">
        <p class="text-xs font-black text-green-600 uppercase tracking-[0.2em] mb-3 animate-[fadeUp_0.5s_ease]">BeejRakshak Mandi Intelligence</p>
        <h2 class="text-4xl md:text-5xl lg:text-[56px] font-black leading-[1.08] text-slate-900 tracking-tight animate-[fadeUp_0.6s_ease]">
          Best Mandi Starts<br />
          with a <em class="not-italic bg-gradient-to-r from-green-600 to-emerald-500 bg-clip-text text-transparent">Smart Deal</em>
        </h2>
        <p class="mt-4 text-[#8b7d6b] text-base md:text-lg leading-relaxed max-w-xl mx-auto animate-[fadeUp_0.7s_ease]">
          ML-powered price predictions & profit-optimised recommendations
          for Gujarat farmers. Know <strong class="text-slate-700">where</strong> & <strong class="text-slate-700">when</strong> to sell.
        </p>

        <!-- CTA Button -->
        <div class="mt-6 animate-[fadeUp_0.8s_ease]">
          <a href="#form-section" 
            class="inline-flex items-center gap-2 bg-gradient-to-br from-green-500 to-green-600 text-white text-sm font-extrabold tracking-wide px-7 py-3.5 rounded-full shadow-lg shadow-green-600/25 hover:shadow-xl hover:shadow-green-600/35 hover:-translate-y-0.5 transition-all duration-300">
            <span>🔍</span> Find Best Mandi
          </a>
        </div>

        <!-- Stat Pills (concise, below CTA) -->
        <div class="flex flex-wrap gap-2 justify-center mt-6 animate-[fadeUp_0.9s_ease]">
          <div class="bg-white/60 backdrop-blur-sm border border-[#dcd1c1] rounded-full px-4 py-1.5 text-xs font-bold text-slate-700 shadow-sm">🤖 XGBoost Regressor</div>
          <div class="bg-white/60 backdrop-blur-sm border border-[#dcd1c1] rounded-full px-4 py-1.5 text-xs font-bold text-slate-700 shadow-sm">📈 MAPE ~1.62%</div>
          <div class="bg-white/60 backdrop-blur-sm border border-[#dcd1c1] rounded-full px-4 py-1.5 text-xs font-bold text-slate-700 shadow-sm">🗓️ 2 Years Data</div>
        </div>
      </div>
    </div>

    <!-- ═══════════════════════════════════════════════════ -->
    <!--   HANDSHAKE VIDEO — Full-bleed, scroll-scrubbed   -->
    <!-- ═══════════════════════════════════════════════════ -->
    <div id="video-section" class="w-full bg-[#e8e2d8] z-0 scrub-section h-[80vh] md:h-[100vh] flex flex-col items-center justify-center">
      <div class="relative w-full h-full overflow-hidden">
        <!-- The scroll-scrubbed video -->
        <video
          ref="handshakeVideo"
          src="/videos/final.mp4"
          class="w-full h-full object-cover mix-blend-darken"
          muted
          playsinline
          preload="auto"
          disablepictureinpicture
          aria-label="Farmer handshake showcase"
        ></video>

        <!-- Vignette fades into page background -->
        <div class="absolute inset-0 pointer-events-none z-20 vignette-overlay"></div>
      </div>
    </div>

  </section>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

const handshakeVideo = ref<HTMLVideoElement | null>(null)
let scrollTriggerInstance: globalThis.ScrollTrigger | null = null

onMounted(() => {
  gsap.registerPlugin(ScrollTrigger)

  const video = handshakeVideo.value
  if (!video) return

  const setupGSAP = () => {
    scrollTriggerInstance = ScrollTrigger.create({
      trigger: '.scrub-section',
      start: 'top bottom',
      end: '75% center',
      scrub: 0.1,
      onUpdate: (self) => {
        if (video.duration) {
          video.currentTime = video.duration * self.progress
        }
      }
    })
  }

  if (video.readyState >= 1) {
    setupGSAP()
  } else {
    video.onloadedmetadata = setupGSAP
  }
})

onUnmounted(() => {
  if (scrollTriggerInstance) {
    scrollTriggerInstance.kill()
  }
})
</script>

<style scoped>
/* Blends the video smoothly into the beige background */
.vignette-overlay {
  background:
    linear-gradient(to bottom, #e8e2d8 0%, transparent 15%, transparent 65%, #e8e2d8 100%);
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
