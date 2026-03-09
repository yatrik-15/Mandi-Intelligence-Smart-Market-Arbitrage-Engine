<template>
  <div class="bg-white border border-[#dcd1c1] rounded-3xl p-8 w-full max-w-2xl shadow-[0_12px_48px_rgba(200,190,170,0.5)]">
    
    <div class="mb-5">
      <label class="block text-xs font-bold text-[#8b7d6b] uppercase tracking-wider mb-2">Select Crop</label>
      <div class="grid grid-cols-3 gap-3">
        <button 
          v-for="crop in crops" :key="crop.name"
          @click="selectedCrop = crop.name"
          :class="[
            'bg-[#f5f0e6] border-[1.5px] rounded-xl py-3 px-2 text-center transition-all text-sm font-bold',
            selectedCrop === crop.name 
              ? 'border-green-600 bg-green-600/10 shadow-[0_0_0_3px_rgba(22,163,74,0.12),0_0_20px_rgba(22,163,74,0.15)] text-slate-800' 
              : 'border-[#dcd1c1] hover:border-green-500 hover:bg-green-500/5 text-slate-600'
          ]"
        >
          <span class="text-3xl block mb-1">{{ crop.icon }}</span>
          {{ crop.name }}
        </button>
      </div>
    </div>

    <div class="mb-5">
      <label class="block text-xs font-bold text-[#8b7d6b] uppercase tracking-wider mb-2">Quantity (kg)</label>
      <input 
        v-model="quantity"
        type="number" 
        placeholder="e.g. 500" 
        min="1" 
        class="w-full bg-[#f5f0e6] border border-[#dcd1c1] rounded-xl text-slate-800 font-sans text-base py-3 px-4 outline-none transition-colors focus:border-green-600 focus:shadow-[0_0_0_3px_rgba(22,163,74,0.12)] placeholder:text-[#8b7d6b]/70"
      />
    </div>

    <button 
      @click="submit"
      :disabled="loading"
      class="w-full py-4 mt-2 bg-gradient-to-br from-green-500 to-green-600 border-none rounded-xl text-white font-extrabold tracking-wide transition-all shadow-[0_4px_20px_rgba(34,197,94,0.35)] flex items-center justify-center gap-2 hover:opacity-90 hover:-translate-y-px hover:shadow-[0_8px_28px_rgba(34,197,94,0.45)] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none"
    >
      <template v-if="loading">
        <div class="w-5 h-5 border-[3px] border-white/30 border-t-white rounded-full animate-spin"></div>
        <span>Analysing all mandis…</span>
      </template>
      <template v-else>
        <span>🔍 Get Best Recommendation</span>
      </template>
    </button>

    <div v-if="error" class="mt-4 bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm">
      ❌ {{ error }}
    </div>

  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits(['result'])
const config = useRuntimeConfig()

const crops = [
  { name: 'Onion', icon: '🧅' },
  { name: 'Tomato', icon: '🍅' },
  { name: 'Potato', icon: '🥔' }
]

const selectedCrop = ref<string | null>(null)
const quantity = ref<number | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

async function submit() {
  if (!selectedCrop.value) {
    error.value = 'Please select a crop first.'
    return
  }
  if (!quantity.value || quantity.value <= 0) {
    error.value = 'Please enter a valid quantity.'
    return
  }

  error.value = null
  loading.value = true
  emit('result', null) // clear previous

  try {
    const res = await fetch(`${config.public.apiUrl}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        crop: selectedCrop.value, 
        quantity: quantity.value, 
        farmer_location: 'Gandhinagar' 
      })
    })

    if (!res.ok) {
      const errData = await res.json()
      throw new Error(errData.detail || 'Server error occurred')
    }

    const data = await res.json()
    emit('result', data)
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}
</script>
