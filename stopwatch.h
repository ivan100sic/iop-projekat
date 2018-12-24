#include <chrono>
#include <iostream>

struct stopwatch {
	typedef std::chrono::time_point<std::chrono::high_resolution_clock> tp;

	tp last_tick;
	bool notock = false;

	void tick() {
		last_tick = std::chrono::high_resolution_clock::now();
	}

	void tock() const {
		std::chrono::duration<double> dur =
			std::chrono::high_resolution_clock::now() - last_tick;
		std::cerr << "Time: " << dur.count() << '\n';
	}

	stopwatch() {
		tick();
	}

	stopwatch(int) {
		tick();
		notock = true;
	}

	~stopwatch() {
		if (!notock) {
			tock();
		}
	}
};