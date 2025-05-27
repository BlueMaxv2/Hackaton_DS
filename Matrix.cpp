#include "Matrix.h"
#include<ostream>
#include<iostream>
using namespace std;

Matrix::Matrix() {
	Matrix::m_1 = nullptr;
	Matrix::m_col = 0;
	Matrix::m_row = 0;
};
Matrix::~Matrix() 
{
	if (m_1 != nullptr) {
		for (int i = 0; i < m_row; ++i)
			delete[] m_1[i];
		delete[] m_1;
		m_1 = nullptr;
	}
};

Matrix::Matrix(const Matrix& matrix) : m_col(matrix.m_col), m_row(matrix.m_row) {
	m_1 = new double* [m_row];
	for (int i = 0; i < m_row; ++i) {
		m_1[i] = new double[m_col];
		for (int j = 0; j < m_col; ++j) {
			m_1[i][j] = matrix.m_1[i][j];
		}
	}
}

Matrix& Matrix::operator=(const Matrix& matrix) {
	if (this != &matrix) {
		// Освобождаем старую память
		if (m_1) {
			for (int i = 0; i < m_row; ++i) delete[] m_1[i];
			delete[] m_1;
		}

		// Копируем данные
		m_col = matrix.m_col;
		m_row = matrix.m_row;
		m_1 = new double* [m_row];
		for (int i = 0; i < m_row; ++i) {
			m_1[i] = new double[m_col];
			for (int j = 0; j < m_col; ++j) {
				m_1[i][j] = matrix.m_1[i][j];
			}
		}
	}
	return *this;
}

void Matrix::MakeMatrix(int col, int row)
{
	double el;
	m_col = col;
	m_row = row;

	m_1 = new double* [row];
	for (int i = 0; i < row; ++i)
		m_1[i] = new double[col];

	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			cout << "Введите " << j + 1 << " элемент на " << i+1 << " строке: ";
			cin >> el;
			m_1[i][j]=el;
		}
	}
}
void PrintMatrix(const Matrix& matrix) {
	for (int i = 0; i < matrix.m_row; ++i) {
		for (int j = 0; j < matrix.m_col; ++j) {
			cout << matrix.m_1[i][j] << " ";
		}
		cout << endl;
	}
}
Matrix Matrix ::sum(const Matrix& matrix) const {
	if (m_col != matrix.m_col || m_row != matrix.m_row) {
		cout << "Матрицы разного размера, попробуйте еще раз" << endl << endl;;
		throw invalid_argument("Матрицы разного размера!");
	}
	Matrix Sum_Matrix;
	Sum_Matrix.m_col = m_col;
	Sum_Matrix.m_row = m_row;

	Sum_Matrix.m_1 = new double* [m_row];
	for (int i = 0; i < m_row; i++) {
		Sum_Matrix.m_1[i] = new double[m_col];

		for (int j = 0; j < m_col; j++) {
			Sum_Matrix.m_1[i][j] = m_1[i][j] + matrix.m_1[i][j];
		}
	}
	return Sum_Matrix;
}
Matrix Matrix::dif(const Matrix& matrix) const {
	if (m_col != matrix.m_col || m_row != matrix.m_row) {
		cout << "Матрицы разного размера, попробуйте еще раз" << endl << endl;;
		throw invalid_argument("Матрицы разного размера!");
	}
	Matrix Dif_Matrix;
	Dif_Matrix.m_col = m_col;
	Dif_Matrix.m_row = m_row;

	Dif_Matrix.m_1 = new double* [m_row];
	for (int i = 0; i < m_row; i++) {
		Dif_Matrix.m_1[i] = new double[m_col];

		for (int j = 0; j < m_col; j++) {
			Dif_Matrix.m_1[i][j] = m_1[i][j] - matrix.m_1[i][j];
		}
	}
	return Dif_Matrix;
}
Matrix Matrix::mult(double el) {

	Matrix Mult_Matrix;
	Mult_Matrix.m_col = m_col;
	Mult_Matrix.m_row = m_row;

	Mult_Matrix.m_1 = new double* [m_row];
	for (int i = 0; i < m_row; i++) {
		Mult_Matrix.m_1[i] = new double[m_col];

		for (int j = 0; j < m_col; j++) {
			Mult_Matrix.m_1[i][j] = m_1[i][j] *el;
		}
	}
	return Mult_Matrix;
}
Matrix Matrix::div(double el) {

	Matrix Div_Matrix;
	Div_Matrix.m_col = m_col;
	Div_Matrix.m_row = m_row;

	Div_Matrix.m_1 = new double* [m_row];
	for (int i = 0; i < m_row; i++) {
		Div_Matrix.m_1[i] = new double[m_col];

		for (int j = 0; j < m_col; j++) {
			Div_Matrix.m_1[i][j] = m_1[i][j] / el;
		}
	}
	return Div_Matrix;
}
