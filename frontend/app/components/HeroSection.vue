<template>
  <section class="w-full flex flex-col items-center">
    
    <!-- Top Text Section (Matches Page Background) -->
    <div class="w-full relative z-10 py-16 px-4 flex flex-col items-center">
      <div class="max-w-3xl text-center">
        <h2 class="text-4xl md:text-5xl lg:text-[54px] font-black leading-tight text-slate-900 tracking-tight">
          Best Mandi Starts<br />
          with a <em class="not-italic text-green-600">Smart Deal</em>
        </h2>
        <p class="mt-4 text-slate-600 text-lg md:text-xl leading-relaxed">
          ML-powered price predictions & profit-optimised recommendations<br class="hidden md:block"/>
          for Gujarat farmers. Know where & when to sell.
        </p>

        <div class="flex flex-wrap gap-2 justify-center mt-6">
          <div class="bg-green-600/10 border border-green-600/30 rounded-full px-4 py-1.5 text-xs font-bold text-green-700">🤖 XGBoost ML</div>
          <div class="bg-green-600/10 border border-green-600/30 rounded-full px-4 py-1.5 text-xs font-bold text-green-700">📈 ~97.6% Accuracy</div>
          <div class="bg-green-600/10 border border-green-600/30 rounded-full px-4 py-1.5 text-xs font-bold text-green-700">🗓️ 4 Years Data</div>
          <div class="bg-green-600/10 border border-green-600/30 rounded-full px-4 py-1.5 text-xs font-bold text-green-700">⚡ ₹0.40/kg Error</div>
        </div>
      </div>
    </div>

    <!-- Handshake Animation Stage (Beige Background) -->
    <div class="w-full bg-[#e8e2d8] z-0 scrub-section pb-16 h-[70vh] md:h-[90vh] flex flex-col items-center justify-center">
      <div class="relative w-full h-full overflow-hidden">
      <!-- The precise scroll-scrubbed video tag -->
      <video 
        ref="handshakeVideo"
        src="/videos/final.mp4" 
        class="w-full h-full object-cover mix-blend-darken" 
        muted 
        playsinline 
        preload="auto" 
        disablepictureinpicture 
        aria-label="Interactive video showcase"
      ></video>
      
        <!-- Edge fades (vignette) to blend nicely into the beige -->
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
      start: 'top bottom', // Start playing when the top of the video enters the bottom of the screen
      end: '75% center',   // Finish early so the video completes before scrolling fully past it
      scrub: 0.1,          // Very tight scrub so it feels instantly connected to the scrollbar
      onUpdate: (self) => {
        if (video.duration) {
          // Calculate the target time based on scroll progress (0 to 1)
          const scrollPos = self.progress;
          // Apply progress to the video duration
          const videoCurrentTime = video.duration * scrollPos;
          
          video.currentTime = videoCurrentTime;
        }
      }
    });
  }

  // If metadata is already loaded (e.g., video is cached), run setup immediately.
  // readyState >= 1 means HAVE_METADATA
  if (video.readyState >= 1) {
    setupGSAP()
  } else {
    // Otherwise wait for the event
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
/* Gradient fades strictly matched to the beige background colour (#e8e2d8) */
.vignette-overlay {
  background: 
    linear-gradient(to bottom, #e8e2d8 0%, transparent 18%, transparent 60%, #e8e2d8 100%);
}
</style>
