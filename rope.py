import carb
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema
import omni.physxdemos as demo
from omni.physx.scripts import physicsUtils
import omni.physx.bindings._physx as physx_bindings
import random
import numpy as np
import isaacsim.core.utils.prims as prim_utils

class RopeFactory:
    """
    Adapted from Isaac Sim Rope demo.
    """
    def create(self, prim_path: str, stage, rope_length):
        self._stage = stage
        self._defaultPrimPath = Sdf.Path(prim_path)
 
        self._linkHalfLength = 0.01 # smaller value makes it smoother
        self._linkRadius = 0.005
        self._ropeLength = rope_length
        self._ropeSpacing = 1.50
        self._coneAngleLimit = 110
        self._rope_damping = 10.0
        self._rope_stiffness = 1.0

        self._capsuleZ = 0.0
        
        # physics options:
        self._contactOffset = 2.0
        self._physicsMaterialPath = self._defaultPrimPath.AppendChild("PhysicsMaterial")
        print(self._defaultPrimPath)
        print(self._physicsMaterialPath)
        UsdShade.Material.Define(self._stage, self._physicsMaterialPath)
        material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(self._physicsMaterialPath))
        material.CreateStaticFrictionAttr().Set(0.5)
        material.CreateDynamicFrictionAttr().Set(0.5)
        material.CreateRestitutionAttr().Set(0)
        return self._createRopes()

    def _createCapsule(self, path: Sdf.Path):
        capsuleGeom = UsdGeom.Capsule.Define(self._stage, path)
        capsuleGeom.CreateHeightAttr(self._linkHalfLength)
        capsuleGeom.CreateRadiusAttr(self._linkRadius)
        capsuleGeom.CreateAxisAttr("X")

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        massAPI.CreateDensityAttr().Set(0.00005)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        physxCollisionAPI.CreateContactOffsetAttr().Set(self._contactOffset)
        physicsUtils.add_physics_material_to_prim(self._stage, capsuleGeom.GetPrim(), self._physicsMaterialPath)

    def _createJoint(self, jointPath):        
        joint = UsdPhysics.Joint.Define(self._stage, jointPath)

        # locked DOF (lock - low is greater than high)
        d6Prim = joint.GetPrim()
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "rotX")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)

        # Moving DOF:
        dofs = ["rotY", "rotZ"]
        for d in dofs:
            limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, d)
            limitAPI.CreateLowAttr(-self._coneAngleLimit)
            limitAPI.CreateHighAttr(self._coneAngleLimit)

            # joint drives for rope dynamics:
            driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, d)
            driveAPI.CreateTypeAttr("force")
            driveAPI.CreateDampingAttr(self._rope_damping)
            driveAPI.CreateStiffnessAttr(self._rope_stiffness)

    def _createRopes(self):
        linkLength = 2.0 * self._linkHalfLength - self._linkRadius
        numLinks = int(self._ropeLength / linkLength)
        
        scopePath = self._defaultPrimPath.AppendChild(f"Rope")
        base = UsdGeom.Xform.Define(self._stage, scopePath)
        base.AddTranslateOp().Set(value=(0.5, 0.5, 0))
        print(scopePath)
        # capsule instancer
        instancerPath = scopePath.AppendChild("rigidBodyInstancer")
        rboInstancer = UsdGeom.PointInstancer.Define(self._stage, instancerPath)
        
        capsulePath = instancerPath.AppendChild("capsule")
        self._createCapsule(capsulePath)
        
        meshIndices = []
        positions = []
        orientations = []

        z = self._capsuleZ + self._linkRadius
        angle_step = 0.15
        angle = 0.0
        radius = 10
        final_angle = 2*(np.random.uniform()-0.5)*3.14/2
        angles = np.linspace(0, final_angle, numLinks+1) 
        for _ in range(2):
            mean = np.random.uniform()
            std = np.random.uniform(0.5, 0.8)
            angles *= (1 + np.exp(-(np.linspace(0, 1, numLinks+1)-mean)**2 / std**2))
        x_values = np.cumsum(linkLength * np.cos(angles))
        y_values = np.cumsum(linkLength * np.sin(angles))
        
        center_x, center_y = np.mean(x_values), np.mean(y_values)
        center_x += np.random.uniform()*0.05
        center_y += np.random.uniform()*0.05
        x_values -= center_x
        y_values -= center_y
        
        angle = 0.0
        for x, y, angle in zip(x_values, y_values, angles):
            meshIndices.append(0)
            tangent = Gf.Vec3f(-np.sin(angle), np.cos(angle), 0.0)  # derivative of curve
            up = Gf.Vec3f(0.0, 0.0, 1.0)

            positions.append(Gf.Vec3f(x, y, z))
            rotation = Gf.Rotation(Gf.Vec3d(0,0,1), angle/3.14*180)
            orientations.append(Gf.Quath(rotation.GetQuat()))
        meshList = rboInstancer.GetPrototypesRel()
        # add mesh reference to point instancer
        meshList.AddTarget(capsulePath)

        rboInstancer.GetProtoIndicesAttr().Set(meshIndices)
        rboInstancer.GetPositionsAttr().Set(positions)
        rboInstancer.GetOrientationsAttr().Set(orientations)
        
        # joint instancer
        jointInstancerPath = scopePath.AppendChild("jointInstancer")
        jointInstancer = PhysxSchema.PhysxPhysicsJointInstancer.Define(self._stage, jointInstancerPath)
        
        jointPath = jointInstancerPath.AppendChild("joint")
        self._createJoint(jointPath)

        meshIndices = []
        body0s = []
        body0indices = []
        localPos0 = []
        localRot0 = []
        body1s = []
        body1indices = []
        localPos1 = []
        localRot1 = []      
        body0s.append(instancerPath)
        body1s.append(instancerPath)

        jointX = self._linkHalfLength - 0.5 * self._linkRadius
        for linkInd in range(numLinks - 1):
            meshIndices.append(0)
            
            body0indices.append(linkInd)
            body1indices.append(linkInd + 1)
                        
            localPos0.append(Gf.Vec3f(jointX, 0, 0)) 
            localPos1.append(Gf.Vec3f(-jointX, 0, 0)) 
            localRot0.append(Gf.Quath(1.0))
            localRot1.append(Gf.Quath(1.0))

        meshList = jointInstancer.GetPhysicsPrototypesRel()
        meshList.AddTarget(jointPath)

        jointInstancer.GetPhysicsProtoIndicesAttr().Set(meshIndices)

        jointInstancer.GetPhysicsBody0sRel().SetTargets(body0s)
        jointInstancer.GetPhysicsBody0IndicesAttr().Set(body0indices)
        jointInstancer.GetPhysicsLocalPos0sAttr().Set(localPos0)
        jointInstancer.GetPhysicsLocalRot0sAttr().Set(localRot0)

        jointInstancer.GetPhysicsBody1sRel().SetTargets(body1s)
        jointInstancer.GetPhysicsBody1IndicesAttr().Set(body1indices)
        jointInstancer.GetPhysicsLocalPos1sAttr().Set(localPos1)
        jointInstancer.GetPhysicsLocalRot1sAttr().Set(localRot1)

        return prim_utils.get_prim_at_path(scopePath)

#stage = omni.usd.get_context().get_stage()
#dem = RopeFactory()
#print(dem.create(stage, 1))