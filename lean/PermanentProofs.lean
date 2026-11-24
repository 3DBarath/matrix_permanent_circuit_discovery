/-
  Lean 4 + Mathlib4 Formalization: Matrix Permanent Degree-2 Row Expansion
  Proves that if row 0 of an (n+1)x(n+1) matrix A has at most two nonzero entries
  at columns j₀ ≠ j₁, then the cofactor sum over row 0 collapses to exactly two terms.
-/
import Mathlib.LinearAlgebra.Matrix.Permanent

open Matrix Finset

/-- Degree-2 row cofactor collapse: any row-0 cofactor sum with at most two nonzero
    entries at j₀ ≠ j₁ reduces to the two corresponding minor terms. -/
theorem perm_degree2_row_cofactor_collapse
    {R : Type*} [CommRing R] {n : ℕ}
    (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R)
    (j₀ j₁ : Fin (n + 1)) (hne : j₀ ≠ j₁)
    (hdeg2 : ∀ j : Fin (n + 1), j ≠ j₀ → j ≠ j₁ → A 0 j = 0) :
    (∑ j : Fin (n + 1), A 0 j * (A.submatrix Fin.succ j.succAbove).permanent) =
      A 0 j₀ * (A.submatrix Fin.succ j₀.succAbove).permanent +
      A 0 j₁ * (A.submatrix Fin.succ j₁.succAbove).permanent := by
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

/-- Essential variable elimination: if `9 ≤ 2 * M - W` for natural numbers `M` and `W`,
    then `M ≥ 5`. -/
theorem essential_var_eliminates_le_five (M W : ℕ) (h : 9 ≤ 2 * M - W) : M ≥ 5 := by
  omega

/-- Essential variable weight bound at `M = 6`: if `9 ≤ 2 * 6 - W`, then `W ≤ 3`. -/
theorem essential_var_m6_forces_w_le_3 (W : ℕ) (h : 9 ≤ 2 * 6 - W) : W ≤ 3 := by
  omega

/-- Degree-1 peeling row cofactor collapse: any row-0 cofactor sum with at most one
    nonzero entry at `j₀` reduces to the single corresponding minor term. -/
theorem deg1_peeling_row_cofactor
    {R : Type*} [CommRing R] {n : ℕ}
    (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R)
    (j₀ : Fin (n + 1))
    (hdeg1 : ∀ j : Fin (n + 1), j ≠ j₀ → A 0 j = 0) :
    (∑ j : Fin (n + 1), A 0 j * (A.submatrix Fin.succ j.succAbove).permanent) =
      A 0 j₀ * (A.submatrix Fin.succ j₀.succAbove).permanent := by
  rw [← Finset.add_sum_erase (Finset.univ : Finset (Fin (n + 1))) _ (Finset.mem_univ j₀)]
  have h_zero : ∑ j ∈ (Finset.univ : Finset (Fin (n + 1))).erase j₀,
      A 0 j * (A.submatrix Fin.succ j.succAbove).permanent = 0 := by
    apply Finset.sum_eq_zero
    intro j hj
    have hj0 : j ≠ j₀ := (Finset.mem_erase.mp hj).1
    rw [hdeg1 j hj0, zero_mul]
  rw [h_zero, add_zero]
