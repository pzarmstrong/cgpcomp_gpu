#include "stdafx.h"
#include "RunTimer.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <string>

clock_t t;

using namespace std;

bool RunTimer::startTimer()
{
	t = clock();
	printf("\n #START# Starting program timer.\n");

	return true;
}

bool RunTimer::startTimer(std::string message)
{
	t = clock();
	cout << "\n #START# Starting " << message << " timer.\n" << endl;

	return true;
}

bool RunTimer::endTimer()
{
	t = clock() - t;
	cout << "\n #END# Timer run time: " << t << " (" << ((long double)t) / CLOCKS_PER_SEC << " seconds).\n\n" << endl;
	t = 0;
	return true;
}

bool RunTimer::endTimer(std::string message)
{
	t = clock() - t;
	cout << "\n #END# " << message << " run time: " << t << " (" << ((long double)t) / CLOCKS_PER_SEC << " seconds).\n\n" << endl;
	t = 0;
	return true;
}

