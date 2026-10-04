# LIBERO-Plus licensed-assets camera compatibility notices

This private, operator-only image is a narrowly scoped derivative for
`libero-plus-licensed-assets-camera-compatibility`. It runs an allowlisted
scene from the author-published `Sylvest/LIBERO-plus` asset archive through the
original MIT LIBERO `TableArena` and MuJoCo renderer. It is **not** a
LIBERO-Plus benchmark image, policy image, task suite, or robot-control image.

## Author-published asset archive

- Repository: `Sylvest/LIBERO-plus`
- Revision: `dd2bd61b7d9a6fef1abc52d606e983b41886a149`
- Asset: `assets.zip`, SHA-256
  `96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`
- Asset card and declared license: [MIT](https://huggingface.co/datasets/Sylvest/LIBERO-plus/blob/dd2bd61b7d9a6fef1abc52d606e983b41886a149/README.md)
- Selected member: `assets/scenes/libero_tabletop_base_style.xml`, SHA-256
  `5e69f8568bedf4a71641fcb62285182d0f6dbe498ea18adad86a13706558033f`

The image contains neither that archive nor the selected scene. It obtains the
archive at runtime by its immutable revision and verifies the full archive hash
before emitting only the allowlisted member to run-scoped storage.

## Original native executor

- Source: `Lifelong-Robot-Learning/LIBERO`
- Revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- License: [MIT](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/LICENSE),
  SHA-256 `e2885fd30a08381b799c4a33385522b23d637b4051b8f9a7f9f2519944b68ff6`
- Native component: `libero.libero.envs.arenas.table_arena.TableArena`

Credit the original LIBERO project and its authors for the native scene executor.
This derivative uses the operator's already inspected private OpenWAM runtime as
its base and preserves its notices; it does not claim either project as NPA
research.

## Explicitly excluded source boundary

The `sylvestf/LIBERO-plus` source revision
`4976dc30028e805ff8094b55501d532c48fec182` has no reviewed LICENSE, NOTICE, or
COPYING declaration. It is not copied, fetched, imported, or executed by this
image or workflow. The MIT declaration for the separate asset archive does not
grant rights for that source. No new NPA EULA, `ACCEPT_*` variable, telemetry
consent, or duplicate attestation is introduced.
