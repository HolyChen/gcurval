#pragma once

#include <cassert>
#include <chrono>
#include <sstream>
#include <iostream>
#include <string>
#include <cstring>
#include <functional>
#include <utility>

namespace gcurval
{

template <typename Duration>
constexpr const char* duration_name();

// Time counter
// Args:
//   Clock: clock type, default = std::chrono::steady_clock
template <typename Clock = std::chrono::steady_clock>
struct TimeCounter final
{
    using TimePoint = typename Clock::time_point;

    TimePoint start_tp = Clock::now();

    TimeCounter() = default;

    template <typename Duration = typename std::chrono::milliseconds, typename Rep = double>
    inline std::string tell() const
    {
        std::chrono::duration<Rep, typename Duration::period> diff = Clock::now() - start_tp;

        std::stringstream ss;
        ss << diff.count() << duration_name<Duration>();

        return ss.str();
    }

    template <typename Rep = double>
    inline std::string tell(const char* name, const char* prefix = "", const char* suffix = "") const
    {
        typename Clock::time_point now_tp = Clock::now();

        std::stringstream ss;

        Rep rep;

        if (!strcmp(name, "h"))
        {
            rep = std::chrono::duration<Rep, std::chrono::hours::period>(now_tp - start_tp).count();
        }
        else if (!strcmp(name, "min"))
        {
            rep = std::chrono::duration<Rep, std::chrono::minutes::period>(now_tp - start_tp).count();
        }
        else if (!strcmp(name, "s"))
        {
            rep = std::chrono::duration<Rep, std::chrono::seconds::period>(now_tp - start_tp).count();
        }
        else if (!strcmp(name, "ms"))
        {
            rep = std::chrono::duration<Rep, std::chrono::milliseconds::period>(now_tp - start_tp).count();
        }
        else if (!strcmp(name, "us"))
        {
            rep = std::chrono::duration<Rep, std::chrono::microseconds::period>(now_tp - start_tp).count();
        }
        else if (!strcmp(name, "ns"))
        {
            rep = std::chrono::duration<Rep, std::chrono::nanoseconds::period>(now_tp - start_tp).count();
        }
        else
        {
            rep = std::chrono::duration<Rep, std::chrono::nanoseconds::period>(now_tp - start_tp).count();
        }

        ss << prefix << " " << rep << name << " " << suffix;

        return ss.str();
    }

    void output(std::ostream& os, const std::string& prefix = "") const
    {
        os << prefix << (prefix.length() > 0 ? " " : "") << tell() << std::endl;
    }

    void output(const char* prefix = "") const
    {
        output(std::cout, prefix);
    }

    inline void reset()
    {
        start_tp = std::chrono::steady_clock::now();
    }
};

template <>
constexpr const char* duration_name<std::chrono::hours>()
{
    return "h";
}

template <>
constexpr const char* duration_name<std::chrono::minutes>()
{
    return "min";
}

template <>
constexpr const char* duration_name<std::chrono::seconds>()
{
    return "s";
}

template <>
constexpr const char* duration_name<std::chrono::milliseconds>()
{
    return "ms";
}

template <>
constexpr const char* duration_name<std::chrono::microseconds>()
{
    return "us";
}

template <>
constexpr const char* duration_name<std::chrono::nanoseconds>()
{
    return "ns";
}

}
