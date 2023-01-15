// main.cpp -- Jake Deery, 2018
#include "main.h"

// var declarations
uint32_t startClock;
uint32_t timeCheck;
mpz_t m;
mpz_t s;
mpz_t temp;

int main() {
  // set up locale support
  #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    // no locale support
  #else
    cout.imbue(locale(""));
  #endif

  // welcome shit
  cout << "Lucas-Lehmer GMP application v1" << endl;
  
  startClock = time(NULL);
  for (uint32_t p = 3; p > 0; p++) {
	  if (lucasLehmerTest(p) == true) {
		  cout << "Prime " << p << " is a mersenne prime!" << endl;
	  }

	  timeCheck = time(NULL);
	  if ((timeCheck - startClock) > 600) break;
  }
  
  cout << "All done (" << (timeCheck - startClock) / 60<< " minutes passed)" << endl;

  // all done, wait for keypress
  #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    system("pause");
  #else
    system("read -p 'Press ENTER to continue . . . '");
  #endif
  return 0;
}

// main process
bool lucasLehmerTest(uint32_t primeToTest) {
  // setup gmp
  mpz_init(m);
  mpz_init(s);
  mpz_init(temp);
  mpz_set_ui(m, 0);
  mpz_set_ui(s, 0);
  mpz_set_ui(temp, 0);

  // if the program sees the value to test isn't prime, just return false
  mpz_set_ui(temp, primeToTest);
  if (!mpz_probab_prime_p(temp, primeToTest)) {
	  return false;
  }

  // set s to 4
  mpz_add_ui(s, s, 4);

  // set m to 2
  mpz_add_ui(m, m, 2);
  // square m with p
  mpz_pow_ui(m, m, primeToTest);
  // subtract 1 from m
  mpz_sub_ui(m, m, 1);

  // do the check!
  for (uint32_t i = 0; i < primeToTest - 2; i++) {
    // square s
    mpz_mul(s, s, s);
    // subtract 2
    mpz_sub_ui(s, s, 2);
    // mod with m
    mpz_mod(s, s, m);
  }

  if (mpz_cmp_ui(s, 0) == 0) { // if s == 0
	mpz_clear(m);
	mpz_clear(s);
	mpz_clear(temp);
	return true;
  }
  else {
    mpz_clear(m);
	mpz_clear(s);
	mpz_clear(temp);
    return false;
  }
}
