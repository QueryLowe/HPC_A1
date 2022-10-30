#include <chrono>
#include <cmath>
#include <cstdio>
#include "matrix.h"
using namespace std;

/* Global variables holding the matrix data. To complete this assignment
 * you are requested to only use arrays and access these arrays with
 * subscripts. Do not use pointers.
 */

const int max_n_elements = 131072;
const int max_n_rows = 16384;
const int test_vector_count = 5;

static double values[max_n_elements];
static int col_ind[max_n_elements];
static int row_ptr_begin[max_n_rows];
static int row_ptr_end[max_n_rows];

/* ---------- self-defined variables ---------- */
/* row compressed storage Matrix */
struct Matrix
{
  int col_ind[max_n_elements];
  int row_ptr_begin[max_n_rows]; //  row_ptr_begin[i]= the first index that the entries of row i appear in col_ind[] and values[]
  int row_ptr_end[max_n_rows];   //  row_ptr_begin[i] = the last index that the entries of row i appear in col_ind[] and values[]
  double values[max_n_elements];
};
/* column compressed storage Matrix */
struct Matrix_col
{
  int row_ind[max_n_elements];
  int col_ptr_begin[max_n_rows];
  int col_ptr_end[max_n_rows];
  double values[max_n_elements];
};

// static Matrix A;
// static Matrix P;
// static Matrix_col L;
// static Matrix U;
// static double x[max_n_rows];
// static double b[max_n_rows];
// static double c[max_n_rows];
// static double y[max_n_rows];
// static double x_new[max_n_rows][test_vector_count];

/* ----- declaration of self-defined functions ----- */
void initialize_A(int n_rows, int nnz, Matrix *M);
void initialize_identity_matrix(int n_rows, Matrix *M);
// void initialize_L(int n_rows, int row_ind[], int col_ptr_begin[], int col_ptr_end[], double values[] );
void LU_factorization(int n_rows, int n_cols, int nnz, Matrix *A, Matrix_col *L, Matrix *P, Matrix *U);
void pivot_search(int pivot, int n_rows, Matrix *U, Matrix_col *L, Matrix *P, int &target_row, double &pivot_value);
void exchange_row(int row_1, int row_2, Matrix *M);
void permutation_L(int pivot, int target_row, Matrix_col *L);
void initialize_L_col(int pivot, Matrix_col *L, int &L_nonzero_num);
void elimination(int pivot, double pivot_value, int n_rows, Matrix *U, Matrix_col *L, int &L_nonzero_num);
void mask_operation(int pivot, int row, Matrix *U, double &l, int n_rows);
void add_or_remove_elements(Matrix *M, int n_rows, int row, int mask[], double target_row_values[]);
void initialize_x(int i, int n_rows, double x[]);
void matrix_multiply_vector(Matrix *M, double vector_1[], double vector_2[], int n_rows);
void get_y(Matrix_col *L, double c[], double y[], int n_rows);
void get_solution_x(Matrix *U, double y[], double x_new[][test_vector_count], int x_new_idx, int n_rows);
double error(double x[], double x_new[][test_vector_count], int x_new_idx, int x_n);

/* --------------------- main() --------------------- */
int main(int argc, char **argv)
{
  if (argc != 2)
  {
    fprintf(stderr, "usage: %s <filename>\n", argv[0]);
    return -1;
  }

  int nnz, n_rows, n_cols;
  bool ok(false);

  ok = load_matrix_market(argv[1], max_n_elements, max_n_rows,
                          nnz, n_rows, n_cols,
                          values, col_ind, row_ptr_begin, row_ptr_end);
  if (!ok)
  {
    fprintf(stderr, "failed to load matrix.\n");
    return -1;
  }

  Matrix A;
  Matrix P; 
  Matrix_col L;
  Matrix U;
  double x[max_n_rows];
  double b[max_n_rows];
  double c[max_n_rows];
  double y[max_n_rows];
  double x_new[max_n_rows][test_vector_count];

  /* Load A */
  initialize_A(n_rows, nnz, &A);
  /* Initialize U = A */
  initialize_A(n_rows, nnz, &U);
  /* Initialize P = identity matrix */
  initialize_identity_matrix(n_rows, &P);
  /* As L should use column condensed storage, we don't initialize L here */

  /* For debugging, can be removed when implementation is finished. */
  // dump_nonzeros(n_rows, U.values, U.col_ind, U.row_ptr_begin, U.row_ptr_end);

  auto factorization_start_time = std::chrono::high_resolution_clock::now();

  /* Perform LU factorization here */
  LU_factorization(n_rows, n_cols, nnz, &A, &L, &P, &U);

  auto factorization_end_time = std::chrono::high_resolution_clock::now();
  auto solve_start_time = std::chrono::high_resolution_clock::now();

  /* Compute all 5 solution vectors here */
  for(int i = 0 ; i< test_vector_count ; i++)
  {
    initialize_x(i , n_rows, x);

    /* compute b = Ax */
    matrix_multiply_vector(&A, x, b, n_rows);
    /* compute c = Pb */
    matrix_multiply_vector(&P, b, c, n_rows);
    // compute y, where Ly=c
    get_y(&L, c, y, n_rows);
    // compute x_new, where U * x_new = y
    get_solution_x(&U, y, x_new, i, n_rows);
  }

  auto solve_end_time = std::chrono::high_resolution_clock::now();

  double relative_errors[test_vector_count] = {0};
  /* Compute relative errors here */
  for (int i = 0; i < test_vector_count; i++)
  {
    relative_errors[i] = error(x, x_new, i, n_rows);
  }

  std::chrono::duration<double> factorization_elapsed_time = factorization_end_time - factorization_start_time;
  std::chrono::duration<double> solve_elapsed_time = solve_end_time - solve_start_time;

  /* Print results */
  fprintf(stdout, "%.20f\n", factorization_elapsed_time.count());
  fprintf(stdout, "%.20f\n", solve_elapsed_time.count());
  for (size_t vector_idx = 0; vector_idx < test_vector_count; ++vector_idx)
  {
    fprintf(stdout, "%.20f\n", relative_errors[vector_idx]);
  }

  return 0;
}

/* ------------------ definitions of self-defined functions ------------------ */
/* Initialize a matrix M = A */
void initialize_A(int n_rows, int nnz, Matrix *M)
{
  for (int row_num = 0; row_num < n_rows; row_num++)
  {
    M->row_ptr_begin[row_num] = row_ptr_begin[row_num];
    M->row_ptr_end[row_num] = row_ptr_end[row_num];
  }
  for (int entry_index = 0; entry_index < nnz; entry_index++)
  {
    M->col_ind[entry_index] = col_ind[entry_index];
    M->values[entry_index] = values[entry_index];
  }
}

/* Initialize a identity matrix */
void initialize_identity_matrix(int n_rows, Matrix *M)
{
  for (int row_num = 0; row_num < n_rows; row_num++)
  {
    M->col_ind[row_num] = row_num;
    M->row_ptr_begin[row_num] = row_num;
    M->row_ptr_end[row_num] = row_num;
    M->values[row_num] = 1;
  }
}

/* (Idle) Initilize matrix L, which is an identity matrix with column compressed storage */
// void initialize_L(int n_rows, int row_ind[], int col_ptr_begin[], int col_ptr_end[], double values[] )
// {
//   for (int col_num = 0; col_num < n_rows; col_num++)
//   {
//     row_ind[col_num] = col_num;
//     col_ptr_begin[col_num] = col_num;
//     col_ptr_end[col_num] = col_num;
//     values[col_num] = 1;
//   }
// }

void LU_factorization(int n_rows, int n_cols, int nnz,
                      Matrix *A, Matrix_col *L, Matrix *P, Matrix *U)
{
  int L_nonzero_num = 0; // num of non-zero entries in L

  for (int pivot = 0; pivot < n_rows - 1; pivot++)
  {
    /* 1. Pivot search ( i>=j )*/
    int target_row = pivot;
    double pivot_value = 0.0;
    pivot_search(pivot, n_rows, U, L, P, target_row, pivot_value);

    /* 2. Permutation on U, P, L */
    if (target_row != pivot)
    {
      exchange_row(target_row, pivot, U);
      exchange_row(target_row, pivot, P);

      /* Permutation on L */
      permutation_L(pivot, target_row, L);
    }
    
    /* L[pivot,pivot] == 1 */
    initialize_L_col(pivot, L, L_nonzero_num); 

    /* 3. Elemination on U, construct L */
    elimination(pivot, pivot_value, n_rows, U, L, L_nonzero_num);
  }

  printf("LU success!\n");
}

/* Pivot Search on (pivot, pivot)
 * @param
 * - pivot: row/col num of pivot position
 * - n_rows: num of total rows
 */
void pivot_search(int pivot, int n_rows, Matrix *U, Matrix_col *L, Matrix *P, int &target_row, double &pivot_value)
{
  /* get the row_num to exchange */
  for (int row = pivot; row < n_rows; row++)
  {
    // int idx = U->row_ptr_begin[row];
    /*  when searching for the pivot in the column,
     *  for a usual thinking, only the first non-zero of each row needs to be examed 1.col_index -> 2.value,
     *  as entries in the columns before pivot position have all been eliminated in the previous iterations
     *  ---
     *  HOWEVER !!! Considering for each row i, with row_ptr_begin[i] and row_ptr_end[i] are determined,
     *  but the order of the same row entries in col_ind[] and values[], whose index are between begin[i] and end[i],
     *  are not necessarily ordered by the column index !!!!!!
     *  As a result, all row entries needs to be examed.
     *  ---
     *  (in this case, the entries are ordered by col index, i.e., following the usual thinking,
     *  so we can switch to the fixed statement, which will do the comparision only once in each row, to accelerate)
     */
    for (int idx = U->row_ptr_begin[row]; idx <= U->row_ptr_end[row]; idx++)
    {
      if (U->col_ind[idx] == pivot) // col index should be the same
      {
        if ( fabs(U->values[idx]) > fabs(pivot_value) )
        {
          pivot_value = U->values[idx];
          target_row = row;
        }
        break; // once column index matches => do the comparsion => jump to the next row
      }
    }
  }
  if (pivot_value == 0.0) /* If all elements in the column in lower triangle are 0, LU cannot be factorized */
    fprintf(stderr, "row %d, Pivot value cannot be 0 !\n", pivot);
  // else
    // fprintf(stderr, "row %d, Pivot value %.20f \n", pivot, pivot_value);
}

void exchange_row(int row_1, int row_2, Matrix *M)
{
  int idx_begin = M->row_ptr_begin[row_1];
  M->row_ptr_begin[row_1] = M->row_ptr_begin[row_2];
  M->row_ptr_begin[row_2] = idx_begin;

  int idx_end = M->row_ptr_end[row_1];
  M->row_ptr_end[row_1] = M->row_ptr_end[row_2];
  M->row_ptr_end[row_2] = idx_end;
}

void permutation_L(int pivot, int target_row, Matrix_col *L)
{
  int k = 0;
  while (k <= pivot - 1)
  {
    k++;
    // L[ target_row, k ]  <->  L[ pivot, k]
    int pivot_row_idx = -1;
    int target_row_idx = -1;

    for (int idx = L->col_ptr_begin[k]; idx <= L->col_ptr_end[k]; idx++)
    {
      if( pivot_row_idx !=-1 && target_row_idx != -1) 
      {// entries to exchange have been found
        break;
      }      
      if (L->row_ind[idx] == pivot)
      {
        pivot_row_idx = idx;
        continue;
      }
      if (L->row_ind[idx] == target_row)
      {
        target_row_idx = idx;
        continue;
      }      
    }

    /* exchange the "pointer" */
    if (pivot_row_idx != -1) // (pivot_row, k) != 0
    {
      L->row_ind[pivot_row_idx] = target_row;
    }
    if (target_row_idx != -1) // (target_row, k) != 0
    {
      L->row_ind[target_row_idx] = pivot;
    }
  }
}

/* L[pivot,pivot] = 1 */
void initialize_L_col(int pivot, Matrix_col *L, int &L_nonzero_num)
{  
  L->row_ind[L_nonzero_num] = pivot;
  L->col_ptr_begin[pivot] = L_nonzero_num;
  L->col_ptr_end[pivot] = L_nonzero_num;
  L->values[L_nonzero_num] = 1.0;
  L_nonzero_num++;
}

void elimination(int pivot, double pivot_value, int n_rows, Matrix *U, Matrix_col *L, int &L_nonzero_num)
{
  for (int row = pivot + 1; row < n_rows; row++)
  {
    for( int idx = U->row_ptr_begin[row]  ; idx <= U->row_ptr_end[row] ; idx++)
    {
      if ( U->col_ind[idx] == pivot )
      {
        /* If the entry of the same pivot column == 0, which means the pivot column target has been eliminated,
         * we can skip the row
         */
        if( U->values[idx] == 0.0)
        {
          break; 
        }
        double l = U->values[idx] / pivot_value;
        mask_operation(pivot, row, U, l, n_rows);

        /* once we got l, we could construct L[ pivot+1:n_rows-1 , pivot], which uses column compressed storage*/
        L->row_ind[L_nonzero_num] = row;
        L->values[L_nonzero_num] = l;
        L->col_ptr_end[pivot] = L_nonzero_num;
        L_nonzero_num++;
      }
    }
  }
}

/* Masking operations on the target row */
void mask_operation(int pivot, int row, Matrix *U, double &l, int n_rows)
{
  int mask[n_rows];
  double target_row_values[n_rows]; // a dense boat to store the calculation result of the row to eliminate pivot column

  /* masker operation: scatter */
  for (int j = U->row_ptr_begin[row]; j <= U->row_ptr_end[row]; j++)
  {
    target_row_values[U->col_ind[j]] = U->values[j];
    mask[U->col_ind[j]] = 1;
  }

  /* go though the pivot row (nonzero entries), update mask boat and value boat */
  for (int j = U->row_ptr_begin[pivot]; j <= U->row_ptr_end[pivot]; j++)
  {
    target_row_values[U->col_ind[j]] -= l * U->values[j]; // elimination
    mask[U->col_ind[j]] = (target_row_values[U->col_ind[j]] == 0.0) ? 0 : 1;
  }

  /* gather, write back the new dense boat value */
  add_or_remove_elements(U, n_rows, row, mask, target_row_values);
}

/* add or remove nonzero elements in a Matrix M,
 * modify M->col_index[], M->values[], M->row_ptr_end[], M->row_ptr_begin[],
 * --
 * Attention! Consering permutation is implemented by switching "pointers", row_ptr_end[row] + 1 != row_ptr_begin[row+1] !!!
 *  - to solve this problem, I created new storage arrays for elements adding or removing.
 */
void add_or_remove_elements(Matrix *M, int n_rows, int row, int mask[], double target_row_values[])
{
  double new_values[max_n_elements];
  int new_col_ind[max_n_elements];
  int new_row_ptr_begin[max_n_rows];
  int new_row_ptr_end[max_n_rows];
  int ptr = 0;

  /* copy elements above current row */
  for (int i = 0; i < row ; i++)
  {
    new_row_ptr_begin[i] = ptr;
    for (int j = M->row_ptr_begin[i]; j <= M->row_ptr_end[i]; j++)
    {
      new_values[ptr] = M->values[j];
      new_col_ind[ptr] = M->col_ind[j];
      ptr++;
    }
    new_row_ptr_end[i] = ptr - 1;
  }

  /* copy elements in current row,
     i.e., write back target_row_values[]  */
  new_row_ptr_begin[row] = ptr;
  int inserted = 0;
  for (int k = 0; k < n_rows; k++)
  {
    if (mask[k] == 1)
    {
      new_col_ind[ptr] = k;
      new_values[ptr] = target_row_values[k];
      ptr++;
      inserted++;
    }
  }
  new_row_ptr_end[row] = ptr - 1;
  if (inserted == 0)
  {
    fprintf(stderr, " A full zero row is got! The Matrix cannot be LU factorized! \n");
  }

  /* copy elements after current row */
  for (int i = row + 1; i < n_rows; i++)
  {
    new_row_ptr_begin[i] = ptr;
    for (int j = M->row_ptr_begin[i]; j <= M->row_ptr_end[i]; j++)
    {
      new_values[ptr] = M->values[j];
      new_col_ind[ptr] = M->col_ind[j];
      ptr++;
    }
    new_row_ptr_end[i] = ptr - 1;
  }

  /* Assign the new ordered arrayes to Matrix M */
  for (int idx = 0; idx < ptr; idx++)
  {
    M->col_ind[idx] = new_col_ind[idx];
    M->values[idx] = new_values[idx];
  }
  for (int i = 0; i < n_rows; i++)
  {
    M->row_ptr_begin[i] = new_row_ptr_begin[i];
    M->row_ptr_end[i] = new_row_ptr_end[i];
  }
  // currently, I just set aside the possible index>ptr in the old M->col[] and M->value[]
}

void initialize_x(int i, int n_rows, double x[])
{
  switch (i)
  {
  case 0: // vector x consisting of all ones.
    for (int j = 0; j < n_rows; j++)
    {
      x[j] = 1.0;
    }
    break;
  case 1: // vector x consisting of 0.1 for every element.
    for (int j = 0; j < n_rows; j++)
    {
      x[j] = 0.1;
    }
    break;
  case 2: // vector x consisting of alternating +1, −1 values: +1, −1, +1, −1, +1, −1, . . . .
    for (int j = 0; j < n_rows; j++)
    {
      x[j] = (j % 2 == 0) ? 1 : -1;
    }
    break;
  case 3: // vector x consisting of alternating +5, −5 values.
    for (int j = 0; j < n_rows; j++)
    {
      x[j] = (j % 2 == 0) ? 5 : -5;
    }
    break;
  case 4: // vector x consisting of alternating +100, −100 values.
    for (int j = 0; j < n_rows; j++)
    {
      x[j] = (j % 2 == 0) ? 100 : -100;
    }
    break;
  default:
    break;
  }
}

void matrix_multiply_vector(Matrix *M, double vector_1[], double vector_2[], int n_rows)
{
  for (int i = 0; i < n_rows; i++)
  {
    for (int idx = M->row_ptr_begin[i]; idx <= M->row_ptr_end[i]; idx++)
    {
      vector_2[M->col_ind[idx]] += vector_1[M->col_ind[idx]] * M->values[idx];
    }
  }
}

void get_y(Matrix_col *L, double c[], double y[], int n_rows)
{
  y[0] = c[0];
  for (int i = 1; i < n_rows; i++)
  {
    double accumulate = 0;
    for (int j = 0; j < i; j++)
    {
      for (int idx = L->col_ptr_begin[j]; idx > L->col_ptr_end[j]; idx++)
      {
        if (L->row_ind[idx] == i)
        {
          accumulate += L->values[idx] * y[j];
        }
      }
    }
    y[i] = c[i] - accumulate;
  }
}

void get_solution_x(Matrix *U, double y[], double x_new[][test_vector_count], int x_new_idx, int n_rows)
{
  x_new[n_rows - 1][x_new_idx] = y[n_rows - 1] / U->values[row_ptr_begin[n_rows]];
  for (int i = n_rows - 2; i >= 0; i--)
  {
    double accumulate = 0;
    for (int idx = U->row_ptr_end[i]; idx > U->row_ptr_begin[i]; idx--)
    {
      accumulate += U->values[idx] * x_new[U->col_ind[idx]][x_new_idx];
    }
    x_new[i][x_new_idx] = (y[i] - accumulate) / U->values[row_ptr_begin[i]];
  }
}

double error(double x[], double x_new[][test_vector_count], int x_new_idx, int x_n)
{
  double sum1 = 0.0;
  double sum2 = 0.0;
  for (int i = 0; i < x_n; i++)
  {
    double distance = fabs(x[i] - x_new[i][x_new_idx]);
    sum1 += distance * distance;
    sum2 += x_new[i][x_new_idx] * x_new[i][x_new_idx];
  }
  return sqrt(sum1 / sum2);
}