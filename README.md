# Nebius Physical AI Solutions

Welcome to the Nebius Physical AI repository. This project hosts a collection of solutions for physical artificial intelligence, providing both Nebius-created and community-contributed implementations that enable developers to build, deploy, and integrate physical AI systems.

Nebius Physical AI Solutions brings together best practices, tools, and libraries for developing advanced physical AI applications. Whether you're working with robotics, computer vision, autonomous systems, or other physical AI domains, this repository provides comprehensive resources and reference implementations. Our goal is to foster a collaborative community where developers can share innovations, contribute improvements, and accelerate the adoption of physical AI technologies across industries.

This repository contains production-ready solutions, experimental prototypes, and documentation that support the full lifecycle of physical AI development. From initial prototyping and model training to deployment and monitoring, you'll find everything needed to successfully implement physical AI systems using Nebius services and infrastructure. The project welcomes contributions from the community and encourages collaboration between Nebius and external developers.

## Repository Structure

This repository is organized to support both Nebius and community-created physical AI solutions:

- **Solutions**: Complete implementations and reference architectures for physical AI applications
- **Documentation**: Guides, tutorials, and best practices for building with Nebius physical AI
- **Community Contributions**: Community-maintained projects and extensions

## Getting Started

To get from a fresh clone to the first validated `npa` workbench and BDD100K
pipeline commands, follow [Getting Started](docs/getting-started.md). For a
deeper `npa` CLI walkthrough, see the [npa quickstart](docs/quickstart.md). For
other Nebius Physical AI Solutions, explore the solutions directory and review
the documentation for your specific use case. Each solution includes its own
README with setup instructions and usage examples.

## Reproducing the Demo

To reproduce the Cosmos, Isaac Lab, GR00T, and FiftyOne workbench demo in your
own Nebius project, follow the [8-GPU H200 demo runbook](docs/demo/8gpu-h200.md).

## API Reference

CLI reference pages are generated from the Typer help output in
[docs/cli](docs/cli/README.md).

For browser-based Rerun review workflows, `npa rerun host` and
`npa rerun share` publish `.rrd` recordings through Nebius S3 presigned URLs
for viewing in `app.rerun.io`.

## Architecture

CLI namespace conventions are documented in
[docs/architecture/cli-namespaces.md](docs/architecture/cli-namespaces.md).

## Contributing

We welcome contributions from the community! Whether you're adding new solutions, improving documentation, or reporting issues, your contributions help make Nebius Physical AI better for everyone.

## License

Copyright © 2026 Nebius BV

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

```
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
