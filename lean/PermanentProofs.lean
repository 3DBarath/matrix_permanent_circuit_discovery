/-
  Lean 4 + Mathlib4 Formalization: Matrix Permanent Degree-2 Row Expansion
  Proves that if row 0 of an (n+1)x(n+1) matrix A has at most two nonzero entries
  at columns j₀ ≠ j₁, then perm A = A 0 j₀ * perm (A.submatrix Fin.succ j₀.succAbove)
                                   + A 0 j₁ * perm (A.submatrix Fin.succ j₁.succAbove).
-/
import Mathlib.LinearAlgebra.Matrix.Permanent

open Matrix Finset

theorem perm_degree2_row_expansion
    {R : Type*} [CommRing R] {n : ℕ}
    (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R)
    (j₀ j₁ : Fin (n + 1)) (hne : j₀ ≠ j₁)
    (hdeg2 : ∀ j : Fin (n + 1), j ≠ j₀ → j ≠ j₁ → A 0 j = 0) :
    perm A = A 0 j₀ * perm (A.submatrix Fin.succ j₀.succAbove) +
             A 0 j₁ * perm (A.submatrix Fin.succ j₁.succAbove) := by
  rw [Matrix.perm_eq_sum_submatrix]
  have hj₁_mem : j₁ ∈ (Finset.univ : Finset (Fin (n + 1))).erase j₀ :=
    Finset.mem_erase.mpr ⟨hne.symm, Finset.mem_univ j₁⟩
  rw [← Finset.add_sum_erase (Finset.univ : Finset (Fin (n + 1))) _ (Finset.mem_univ j₀)]
  rw [← Finset.add_sum_erase ((Finset.univ : Finset (Fin (n + 1))).erase j₀) _ hj₁_mem]
  have h_zero : ∑ j ∈ ((Finset.univ : Finset (Fin (n + 1))).erase j₀).erase j₁,
      A 0 j * perm (A.submatrix Fin.succ j.succAbove) = 0 := by
    apply Finset.sum_eq_zero
    intro j hj
    have hj1 : j ≠ j₁ := (Finset.mem_erase.mp hj).1
    have hj0 : j ≠ j₀ := (Finset.mem_erase.mp (Finset.mem_erase.mp hj).2).1
    rw [hdeg2 j hj0 hj1, zero_mul]
  rw [h_zero, add_zero, add_assoc]
