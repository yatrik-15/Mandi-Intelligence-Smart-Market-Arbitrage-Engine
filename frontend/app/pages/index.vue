<template>
  <div class="w-full flex-1 flex flex-col items-center">
    <!-- 1. Hero Handshake Animation -->
    <HeroSection />

    <!-- 2. Main Content (Form + Results) -->
    <div class="w-full flex-1 flex flex-col items-center mt-12 relative z-20 px-4 pb-16">
      <FormCard @result="onResult" />
      
      <div v-if="resultData" class="w-full flex justify-center mt-6">
        <ResultCard :data="resultData" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import HeroSection from '~/components/HeroSection.vue'
import FormCard from '~/components/FormCard.vue'
import ResultCard from '~/components/ResultCard.vue'

const resultData = ref<any>(null)

function onResult(data: any) {
  resultData.value = data
}
</script>
