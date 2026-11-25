/-
  Lean 4 + Mathlib4 Formalization: Matrix Permanent Laplace Expansion,
  Walsh-Hadamard Parity Orthogonality, and Rank-1 Zero-Diagonal Obstruction
-/
import Mathlib.LinearAlgebra.Matrix.Permanent
import Mathlib.Data.Fintype.BigOperators
import Mathlib.GroupTheory.Perm.Fin

open Matrix Finset Equiv

section LaplaceExpansion

/-- Laplacian expansion of the permanent of an `(n+1) × (n+1)` matrix along column `0`. -/
theorem permanent_succ_column_zero {R : Type*} [CommRing R] {n : ℕ}
    (A : Matrix (Fin n.succ) (Fin n.succ) R) :
    A.permanent = ∑ i : Fin n.succ, A i 0 * (A.submatrix i.succAbove Fin.succ).permanent := by
  simp only [Matrix.permanent]
  rw [Finset.univ_perm_fin_succ, ← Finset.univ_product_univ]
  simp only [Finset.sum_map, Equiv.toEmbedding_apply, Finset.sum_product, Matrix.submatrix]
  refine Finset.sum_congr rfl fun i _ => Fin.cases ?_ (fun i => ?_) i
  · simp only [Fin.prod_univ_succ, Finset.mul_sum,
      Equiv.Perm.decomposeFin_symm_apply_zero, Equiv.Perm.decomposeFin_symm_apply_succ,
      Fin.succAbove_zero, Equiv.swap_self, Equiv.coe_refl, id, of_apply]
  · simp only [Fin.prod_univ_succ, Equiv.Perm.decomposeFin_symm_apply_zero,
      Equiv.Perm.decomposeFin_symm_apply_succ, Finset.mul_sum, of_apply]
    refine Fintype.sum_equiv (Equiv.mulLeft i.cycleRange) _ _ fun σ => ?_
    simp only [Equiv.coe_mulLeft, Equiv.Perm.coe_mul, Function.comp_apply, Fin.succAbove_cycleRange]

/-- Laplacian expansion of the permanent of an `(n+1) × (n+1)` matrix along row `0`. -/
theorem permanent_succ_row_zero {R : Type*} [CommRing R] {n : ℕ}
    (A : Matrix (Fin n.succ) (Fin n.succ) R) :
    A.permanent = ∑ j : Fin n.succ, A 0 j * (A.submatrix Fin.succ j.succAbove).permanent := by
  rw [← permanent_transpose A, permanent_succ_column_zero]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← permanent_transpose]
  simp only [transpose_apply, transpose_submatrix, transpose_transpose]

/-- Degree-1 peeling directly on `A.permanent`: if row `0` has at most one nonzero entry at `j₀`,
    then `A.permanent` collapses to `A 0 j₀ * (A.submatrix Fin.succ j₀.succAbove).permanent`. -/
theorem permanent_deg1_peeling
    {R : Type*} [CommRing R] {n : ℕ}
    (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R)
    (j₀ : Fin (n + 1))
    (hdeg1 : ∀ j : Fin (n + 1), j ≠ j₀ → A 0 j = 0) :
    A.permanent = A 0 j₀ * (A.submatrix Fin.succ j₀.succAbove).permanent := by
  rw [permanent_succ_row_zero]
  rw [← Finset.add_sum_erase (Finset.univ : Finset (Fin (n + 1))) _ (Finset.mem_univ j₀)]
  have h_zero : ∑ j ∈ (Finset.univ : Finset (Fin (n + 1))).erase j₀,
      A 0 j * (A.submatrix Fin.succ j.succAbove).permanent = 0 := by
    apply Finset.sum_eq_zero
    intro j hj
    have hj0 : j ≠ j₀ := (Finset.mem_erase.mp hj).1
    rw [hdeg1 j hj0, zero_mul]
  rw [h_zero, add_zero]

/-- Degree-2 row cofactor collapse directly on `A.permanent`: if row `0` has at most two nonzero
    entries at `j₀ ≠ j₁`, then `A.permanent` collapses to the sum of those two minor terms. -/
theorem permanent_deg2_collapse
    {R : Type*} [CommRing R] {n : ℕ}
    (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R)
    (j₀ j₁ : Fin (n + 1)) (hne : j₀ ≠ j₁)
    (hdeg2 : ∀ j : Fin (n + 1), j ≠ j₀ → j ≠ j₁ → A 0 j = 0) :
    A.permanent =
      A 0 j₀ * (A.submatrix Fin.succ j₀.succAbove).permanent +
      A 0 j₁ * (A.submatrix Fin.succ j₁.succAbove).permanent := by
  rw [permanent_succ_row_zero]
  have hj₁_mem : j₁ ∈ (Finset.univ : Finset (Fin (n + 1))).erase j₀ :=
    Finset.mem_erase.mpr ⟨hne.symm, Finset.mem_univ j₁⟩
  rw [← Finset.add_sum_erase (Finset.univ : Finset (Fin (n + 1))) _ (Finset.mem_univ j₀)]
  rw [← Finset.add_sum_erase ((Finset.univ : Finset (Fin (n + 1))).erase j₀) _ hj₁_mem]
  have h_zero : ∑ j ∈ ((Finset.univ : Finset (Fin (n + 1))).erase j₀).erase j₁,
      A 0 j * (A.submatrix Fin.succ j.succAbove).permanent = 0 := by
    apply Finset.sum_eq_zero
    intro j hj
    have hj1 : j ≠ j₁ := (Finset.mem_erase.mp hj).1
    have hj0 : j ≠ j₀ := (Finset.mem_erase.mp (Finset.mem_erase.mp hj).2).1
    rw [hdeg2 j hj0 hj1, zero_mul]
  rw [h_zero, add_zero]

end LaplaceExpansion

section WalshHadamardParity

/-- Sign character on `Bool`: `true ↦ -1`, `false ↦ 1`. -/
def boolSign (R : Type*) [CommRing R] (b : Bool) : R :=
  if b then -1 else 1

/-- `0/1` indicator on `Bool`: `true ↦ 1`, `false ↦ 0`. -/
def boolBit (R : Type*) [CommRing R] (b : Bool) : R :=
  if b then 1 else 0

@[simp]
theorem boolSign_mul_self (R : Type*) [CommRing R] (b : Bool) :
    boolSign R b * boolSign R b = 1 := by
  cases b <;> simp [boolSign]

/-- Walsh-Hadamard parity orthogonality (Theorem 2.1, Eq. 4):
    For any proper subset `S ≠ univ` of `Fin m`, the parity character `σ(y) = ∏ i, boolSign (y i)`
    is orthogonal to `χ_S(y) = ∏ i ∈ S, boolSign (y i)`. -/
theorem walsh_hadamard_parity_orthogonality_zero
    {R : Type*} [CommRing R] {m : ℕ} (S : Finset (Fin m)) (hS : S ≠ Finset.univ) :
    ∑ y : Fin m → Bool,
      (∏ i : Fin m, boolSign R (y i)) * (∏ i ∈ S, boolSign R (y i)) = 0 := by
  have hfactor : ∀ y : Fin m → Bool,
      (∏ i : Fin m, boolSign R (y i)) * (∏ i ∈ S, boolSign R (y i)) =
      ∏ i : Fin m, (if i ∈ S then (1 : R) else boolSign R (y i)) := by
    intro y
    have hprod : ∏ i ∈ S, boolSign R (y i) = ∏ i : Fin m, (if i ∈ S then boolSign R (y i) else 1) := by
      rw [← Finset.prod_filter, Finset.filter_mem_eq_inter, Finset.univ_inter]
    rw [hprod, ← Finset.prod_mul_distrib]
    refine Finset.prod_congr rfl fun i _ => ?_
    by_cases hi : i ∈ S <;> simp [hi]
  simp_rw [hfactor]
  rw [← Fintype.prod_sum (fun i b => if i ∈ S then (1 : R) else boolSign R b)]
  obtain ⟨i₀, _, hi₀⟩ := Finset.not_subset.mp (fun h => hS (Finset.eq_univ_iff_forall.mpr fun i => h (Finset.mem_univ i)))
  exact Finset.prod_eq_zero (Finset.mem_univ i₀) (by simp [hi₀, boolSign])

/-- At `S = univ`, the parity inner product equals `2^m`. -/
theorem walsh_hadamard_parity_orthogonality_univ
    {R : Type*} [CommRing R] {m : ℕ} :
    ∑ y : Fin m → Bool,
      (∏ i : Fin m, boolSign R (y i)) * (∏ i : Fin m, boolSign R (y i)) = (2 : R) ^ m := by
  have hterm : ∀ y : Fin m → Bool,
      (∏ i : Fin m, boolSign R (y i)) * (∏ i : Fin m, boolSign R (y i)) = (1 : R) := by
    intro y
    rw [← Finset.prod_mul_distrib]
    simp
  simp [hterm, Fintype.card_bool]

/-- Monomial parity annihilation (Theorem 2.1(3)):
    Any multilinear monomial `∏ i ∈ T, boolBit R (y i)` of degree `|T| < m`
    vanishes identically against the maximal parity character `σ(y)`. -/
theorem monomial_parity_annihilation
    {R : Type*} [CommRing R] {m : ℕ} (T : Finset (Fin m)) (hT : T.card < m) :
    ∑ y : Fin m → Bool,
      (∏ i : Fin m, boolSign R (y i)) * (∏ i ∈ T, boolBit R (y i)) = 0 := by
  have hfactor : ∀ y : Fin m → Bool,
      (∏ i : Fin m, boolSign R (y i)) * (∏ i ∈ T, boolBit R (y i)) =
      ∏ i : Fin m, (if i ∈ T then boolSign R (y i) * boolBit R (y i) else boolSign R (y i)) := by
    intro y
    have hprod : ∏ i ∈ T, boolBit R (y i) = ∏ i : Fin m, (if i ∈ T then boolBit R (y i) else 1) := by
      rw [← Finset.prod_filter, Finset.filter_mem_eq_inter, Finset.univ_inter]
    rw [hprod, ← Finset.prod_mul_distrib]
    refine Finset.prod_congr rfl fun i _ => ?_
    by_cases hi : i ∈ T <;> simp [hi]
  simp_rw [hfactor]
  rw [← Fintype.prod_sum (fun i b => if i ∈ T then boolSign R b * boolBit R b else boolSign R b)]
  have hne : T ≠ Finset.univ := by
    intro h
    rw [h, Finset.card_univ, Fintype.card_fin] at hT
    exact Nat.lt_irrefl m hT
  have hex : ∃ i₀ : Fin m, i₀ ∉ T := by
    by_contra hall
    push Not at hall
    exact hne (Finset.eq_univ_iff_forall.mpr hall)
  obtain ⟨i₀, hi₀⟩ := hex
  exact Finset.prod_eq_zero (Finset.mem_univ i₀) (by simp [hi₀, boolSign])

end WalshHadamardParity

section RankObstruction

/-- Row-layered rank-1 zero-diagonal obstruction (Proposition 5.1):
    Over any integral domain `R`, a symmetric rank-1 matrix `(i, j) ↦ c * u i * u j`
    whose diagonal vanishes identically (`∀ i, c * u i * u i = 0`) is identically zero. -/
theorem sym_rank1_zero_diag_eq_zero
    {R : Type*} [CommRing R] [IsDomain R] {n : ℕ}
    (c : R) (u : Fin n → R)
    (hdiag : ∀ i : Fin n, c * u i * u i = 0) :
    ∀ i j : Fin n, c * u i * u j = 0 := by
  by_cases hc : c = 0
  · intro i j
    simp [hc]
  · have hu : ∀ i : Fin n, u i = 0 := by
      intro i
      have h := hdiag i
      rw [mul_assoc] at h
      rcases mul_eq_zero.mp h with hc0 | huu
      · exact absurd hc0 hc
      · rcases mul_eq_zero.mp huu with hi | hi <;> exact hi
    intro i j
    simp [hu i]

end RankObstruction
