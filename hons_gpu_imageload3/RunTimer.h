#pragma once

#include <string>

class RunTimer
{
public:
	bool startTimer();
	bool startTimer(std::string);
	bool endTimer();
	bool endTimer(std::string);
};

