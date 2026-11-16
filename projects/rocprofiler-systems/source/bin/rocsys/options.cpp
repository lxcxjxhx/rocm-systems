// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocsys/options.hpp"
#include "rocsys/schema.hpp"

#include "parser/cli_parser.hpp"
#include "parser/configuration_parser.hpp"
#include "parser/help.hpp"
#include "parser/schema.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::rocsys
{
namespace
{
constexpr std::string_view k_every_option = "all";

parser::configuration_parser
make_config()
{
    auto config = parser::configuration_parser{};
    // rocsys reads only its own command line. Each tool reads the shell environment.
    config.set_environment([](std::string_view) { return std::optional<std::string>{}; });
    config.set_unknown_option(parser::unknown_option::forward);
    return config;
}

/// Register the --traces backend on a parser that accepted the catalog. add_backend()
/// refuses only a parser that is not initialized or a taken name, so a refusal is a bug.
void
add_trace_backend(parser::configuration_parser&   config,
                  const std::vector<std::string>& supported_domains)
{
    if(!config.add_backend(make_trace_backend(supported_domains)))
    {
        throw std::logic_error("rocsys could not register its --traces backend");
    }
}
}  // namespace

options
parse_options(const std::vector<std::string>& args,
              const std::vector<std::string>& supported_domains)
{
    auto config = make_config();
    if(!config.initialize(make_schema()))
    {
        return { .environment = {},
                 .tool_args   = {},
                 .command     = {},
                 .errors      = { config.failure() } };
    }
    add_trace_backend(config, supported_domains);
    static_cast<void>(config.parse(args));

    auto environment = config.export_as_environment();
    if(const auto traces = config.get<std::vector<std::string>>("traces"))
    {
        environment.merge(traces_environment(*traces));
    }
    return { .environment = std::move(environment),
             .tool_args   = config.unparsed(),
             .command     = config.command(),
             .errors      = config.errors() };
}

std::string
options_summary()
{
    auto config = make_config();
    return config.initialize(make_schema())
               ? parser::format_summary(*config.config_schema())
               : config.failure();
}

std::optional<std::string>
options_help(std::string_view topic, const std::vector<std::string>& supported_domains)
{
    auto config = make_config();
    if(!config.initialize(make_schema()))
    {
        return config.failure();
    }
    add_trace_backend(config, supported_domains);
    if(topic == k_every_option)
    {
        return config.render_help();
    }
    const auto catalog = config.config_schema();
    if(catalog->params_in_category(topic).empty() &&
       catalog->find_param_by_arg(topic) == nullptr)
    {
        return std::nullopt;
    }
    return config.render_help(parser::param_level::basic, topic);
}
}  // namespace rocprofsys::rocsys
