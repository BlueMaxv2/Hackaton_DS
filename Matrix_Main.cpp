#include <iostream>
#include "Matrix.h"

int main()
{
	setlocale(LC_ALL, "ru");
	int x;
	int col = 0, row = 0;
	while (true)
	{
		cout << "Что Вы хотите сделать?" << endl << "1-вывести матрицу" << endl << "2-сложить матрицы" << endl;
		cout << "3-вычесть матрицы" << endl << "4-умножить матрицу на число" << endl << "5-Разделить матрицу на число" << endl;
		cout << "6-найти обратную матрицу" << endl << "7-найти определить матрицы" << endl << "8-решить СЛАУ" << endl;
		cout << "Любое другое число - выход" << endl;
		cin >> x;
		if (x == 1) 						///вывод матрицы
		{
			cout << "Введите количество строк Вашей матрицы: ";
			cin >> row;
			cout << "Введите количество стобцов Вашей матрицы: ";
			cin >> col;
			Matrix mat;
			mat.MakeMatrix(col, row);
			PrintMatrix(mat);
		}
		if (x == 2)
		{
			cout << "Введите количество строк 1 матрицы: ";
			cin >> row;
			cout << "Введите количество стобцов 1 матрицы: ";
			cin >> col;
			Matrix mat1;
			mat1.MakeMatrix(col, row);

			int row1, col1;
			cout << "Введите количество строк 2 матрицы: ";
			cin >> row1;
			cout << "Введите количество стобцов 2 матрицы: ";
			cin >> col1;
			Matrix mat2;
			mat2.MakeMatrix(col1, row1);
			try {
				Matrix sum = mat1.sum(mat2);
				cout << "Сумма матриц:" << endl;
				PrintMatrix(mat1);
				cout << "+" << endl;
				PrintMatrix(mat2);
				cout << "=" << endl;
				PrintMatrix(sum);
			}
			catch (invalid_argument e) {};
		}
		if (x == 3)
		{
			cout << "Введите количество строк 1 матрицы: ";
			cin >> row;
			cout << "Введите количество стобцов 1 матрицы: ";
			cin >> col;
			Matrix mat1;
			mat1.MakeMatrix(col, row);

			int row1, col1;
			cout << "Введите количество строк 2 матрицы: ";
			cin >> row1;
			cout << "Введите количество стобцов 2 матрицы: ";
			cin >> col1;
			Matrix mat2;
			mat2.MakeMatrix(col1, row1);
			try {
				Matrix dif = mat1.dif(mat2);
				cout << "Сумма матриц:" << endl;
				PrintMatrix(mat1);
				cout << "-" << endl;
				PrintMatrix(mat2);
				cout << "=" << endl;
				PrintMatrix(dif);
			}
			catch (invalid_argument e)
			{
			}
		}
		if (x == 4) 
		{	
			double el = 0;
			cout << "Введите количество строк Вашей матрицы: ";
			cin >> row;
			cout << "Введите количество стобцов Вашей матрицы: ";
			cin >> col;
			cout << "Введит число на которое хотите умножить матрицу: ";
			cin >>el ;
			Matrix mat1;
			mat1.MakeMatrix(col, row);

			Matrix mult = mat1.mult(el);
			cout << "Умножение матрицы на число "<<el<<" :" << endl;
			PrintMatrix(mat1);
			cout << "*" << endl;
			cout << el << endl;
			cout << "=" << endl;
			PrintMatrix(mult);
		}
		if (x == 5)
		{
			double el = 0;
			cout << "Введите количество строк Вашей матрицы: ";
			cin >> row;
			cout << "Введите количество стобцов Вашей матрицы: ";
			cin >> col;
			cout << "Введит число на которое хотите разделить матрицу: ";
			cin >> el;
			Matrix mat1;
			mat1.MakeMatrix(col, row);

			Matrix mult = mat1.div(el);
			cout << "Деление матрицы на число " << el << " :" << endl;
			PrintMatrix(mat1);
			cout << "/" << endl;
			cout << el << endl;
			cout << "=" << endl;
			PrintMatrix(mult);
		}
	}
	return 0;
}

