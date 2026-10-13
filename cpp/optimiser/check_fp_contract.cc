// Configure-time check (cpp/CMakeLists.txt try_run): exit 0 when a*b+c is NOT fused under the
// build's own flags. The product is 1 - 2^-60 exactly; rounded to a double it is 1, so the unfused
// sum is 0, while a fused multiply-add keeps the product exact and gives -2^-60.
int main()
{
    volatile double a = 1.0 + 0x1p-30, b = 1.0 - 0x1p-30, c = -1.0;
    const double x = a, y = b, z = c;
    const double sum = x * y + z;
    return sum == 0.0 ? 0 : 1;
}
