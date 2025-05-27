#pragma once
#include <ostream>
using namespace std;
class Matrix
{
private:
	double** m_1;
	int m_col;
	int m_row;
public:
	Matrix();
	~Matrix();
	void MakeMatrix(int col, int row);
	friend void PrintMatrix(const Matrix& matrix);
	Matrix(const Matrix& other);                       //  онструктор копировани€
	Matrix& operator=(const Matrix& other);           // присваивание 
	Matrix sum(const Matrix& matrix) const;
	Matrix dif(const Matrix& matrix) const;
	Matrix mult(double el);
	Matrix div(double el);
};

