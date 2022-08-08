#ifndef TEMPLATEBUILDER_H
#define TEMPLATEBUILDER_H

#include "Common.cuh"

#include <utility>
#include <array>

#include "Integrate.cuh"
#include "Metropolis.cuh"

namespace TemplateBuilder
{
	// Base template for when no args are given
	// This has no definition so it won't compile
	template<int nNucleonsA, int nNucleonsB, int totalThreads, const auto& a, typename = void>
	struct Make_SAB_Integration_Function_Helper;

	// Force interpretation of std::index_sequence as std::size_t... parameter pack
	// Then specialize the base template for std::index_sequence so this
	// template won't be used again
	template<int nNucleonsA, int nNucleonsB, int totalThreads, const auto& a, std::size_t... i>
	struct Make_SAB_Integration_Function_Helper<nNucleonsA, nNucleonsB, totalThreads, a, std::index_sequence< i... > > {
		constexpr static auto type = MCIntegrate_S_AB<nNucleonsA, nNucleonsB, totalThreads, a.at( i )... >;
	};

	// Just calls the helper to construct the type
	template<int nNucleonsA, int nNucleonsB, int totalThreads, const auto& a>
	struct Make_SAB_Integration_Function {
		constexpr static auto type = Make_SAB_Integration_Function_Helper<nNucleonsA, nNucleonsB, totalThreads, a, std::make_index_sequence<a.size()>>::type;
	};

	// Base template for when no args are given
	// This has no definition so it won't compile
	template<int nNucleonsA, int nNucleonsB, int totalThreads, const auto& a, typename = void>
	struct Make_R2_Integration_Function_Helper;

	// Force interpretation of std::index_sequence as std::size_t... parameter pack
	// Then specialize the base template for std::index_sequence so this
	// template won't be used again
	template<int nNucleonsA, int nNucleonsB, int totalThreads, const auto& a, std::size_t... i>
	struct Make_R2_Integration_Function_Helper<nNucleonsA, nNucleonsB, totalThreads, a, std::index_sequence< i... > > {
		constexpr static auto type = MCIntegrate_R2<nNucleonsA, nNucleonsB, totalThreads, a.at( i )... >;
	};

	// Just calls the helper to construct the type
	template<int nNucleonsA, int nNucleonsB, int totalThreads, const auto& a>
	struct Make_R2_Integration_Function {
		constexpr static auto type = Make_R2_Integration_Function_Helper<nNucleonsA, nNucleonsB, totalThreads, a, std::make_index_sequence<a.size()>>::type;
	};
    
    // Base template for when no args are given
	// This has no definition so it won't compile
	template<int nNucleons, int totalThreads, const auto& a, typename = void>
	struct Make_WarmupMetropolis_Function_Helper;

	// Force interpretation of std::index_sequence as std::size_t... parameter pack
	// Then specialize the base template for std::index_sequence so this
	// template won't be used again
	template<int nNucleons, int totalThreads, const auto& a, std::size_t... i>
	struct Make_WarmupMetropolis_Function_Helper<nNucleons, totalThreads, a, std::index_sequence< i... > > {
		constexpr static auto type = WarmupMetropolis<nNucleons, totalThreads, a.at( i )... >;
	};

	// Just calls the helper to construct the type
	template<int nNucleons, int totalThreads, const auto& a>
	struct Make_WarmupMetropolis_Function {
		constexpr static auto type = Make_WarmupMetropolis_Function_Helper<nNucleons, totalThreads, a, std::make_index_sequence<a.size()>>::type;
	};
    
    // I took this function from somewhere, but I forget where.
	template<class T1,          class T2,
		 size_t S1,         size_t S2,
		 size_t ... SPack1, size_t ... SPack2>
	constexpr std::array<T1, S1 + S2> 
	ConcatArrayHelper(std::array<T1, S1> const arr1, std::array<T2, S2> const arr2,
		    std::integer_sequence<size_t, SPack1...>, std::integer_sequence<size_t, SPack2...>)
	{
		return {{arr1[SPack1]..., arr2[SPack2]...}};										  	
	}
	
    // I took this function from somewhere, but I forget where.
	template<class T1,  class T2,
		 size_t S1, size_t S2>
	constexpr std::array<T1, S1 + S2> 
	ConcatArray(std::array<T1, S1> const arr1, std::array<T2, S2> const arr2)
	{
		static_assert(std::is_same<T1, T2>::value, "Array have different value type");
		return ConcatArrayHelper(arr1, arr2, std::make_index_sequence<S1>{}, std::make_index_sequence<S2>{});
	}
	
    // I took this function from somewhere, but I forget where.
	template<int N, typename T>
	static constexpr std::array<T, N> ConstArray(T val)
	{
		std::array<T, N> a{};
		for (size_t idx = 0; idx < a.size(); ++idx)
		{
			a[idx] = val;
		}
		return a;
	}
    
    // I actually made this function!
    template<int N, typename T>
    static constexpr std::array<T, N> AlternatingArray(T val1, T val2) {
        std::array<T, 2 * N> a{};
        for (size_t idx = 0; idx < a.size(); ++idx)
        {
            a[idx] = ((idx % 2) == 0 ? val1 : val2);    
		}
    }
	
	
	// // Base template
	// template <typename T, const T pdfFunction, const auto& arr, typename = void>
	// struct FunctionSpecializerHelper;
	
	// The following specializations shouldn't be necessary, but nvcc isn't smart enough
	// to allow using `auto` instead of specifying function pointers. So copy paste it is.
	// Really I have no idea why nvcc doesn't allow just using auto since g++ does but whatever.
	
	// // Specialize for 2 float function
	// template <float (*pdfFunction)(float, float), const auto& arr, std::size_t... i>
	// struct FunctionSpecializerHelper<float (*)(float, float), pdfFunction, arr, std::index_sequence< i... > > {
	// 	constexpr static PDF type = [](float r) -> float {return pdfFunction(r, arr.at( i )...); };
	// };
	
	//Specialize for 3 float function
	// template <float (*pdfFunction)(float, float, float), const auto& arr, std::size_t... i>
	// __device__
	// struct FunctionSpecializerHelper<float (*)(float, float, float), pdfFunction, arr, std::index_sequence< i... > > {
	// 	constexpr static PDF type = [] (float r) -> float {return pdfFunction(r, arr.at( i )...); };
	// };
	
	// template <const auto pdfFunction, const auto& arr>
	// struct FunctionSpecializer {
	// 	constexpr static PDF type = FunctionSpecializerHelper<decltype(pdfFunction), pdfFunction, arr, std::make_index_sequence<arr.size()>>::type;
	// };
	
	// struct FloatContainer
	// {
	// 	const float f;
		
	// 	__host__ __device__
	// 	constexpr FloatContainer(float f) : f(f) {}
	// };
	
	// template<const auto pdfFunction, const FloatContainer... floats>
	// struct Functor
	// {	
	// 	__device__ __host__
	// 	constexpr Functor() {}

	// 	__device__ __host__
	// 	static float PDF(float r) { return pdfFunction(r, floats.f...); };
	// };
}

#endif // TEMPLATEBUILDER_H
